#version 460
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : require
#define PI 3.141592

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

uniform int u_frameCount;
uniform int u_seed;

layout(rgba32f, binding = 0) uniform image2D FRAME;

layout(std430, binding = 0) buffer origin {
    vec3 ORIGIN;
};
layout(std430, binding = 2) buffer mousePos {
    vec3 MOUSE_POS;
};
layout(std430, binding = 1) buffer rotation {
    vec3 ROTATION;
};
struct triangle {
    vec3 v1;   // 12 bytes
    vec3 v2;   // 12 bytes
    vec3 v3;   // 12 bytes
    vec3 n1;   // 12 bytes
    vec3 n2;   // 12 bytes
    vec3 n3;   // 12 bytes
    vec3 vt1;  // 8 bytes
    vec3 vt2;  // 8 bytes
    vec3 vt3;  // 8 bytes
    vec3 material; // 4 bytes
};
layout(std430, binding = 3) buffer TriangleData {
    triangle Triangles[];
};
//
layout(std430, binding = 4) buffer Parameters {
    float screenSize;
    float focalLength;
    float resolution;
    float screenHratio;
    float SAMPLE_RES;
    float MAX_BOUNCES;
    float GAMMA;
    float BLUR;
    float FOCAL_DISTANCE;
    float RAYTRACING;
    float DEBUG;
    float AUTO_FOCUS;
};
//
layout(std430, binding = 5) buffer ImplicitData {
    float ImpData[];
};
layout(std430, binding = 7) buffer ellipsoidData {
    float EllipData[];
};
layout(std430, binding = 10) buffer BvhData {
    float BVHdata[];
};
layout(std430, binding = 11) buffer bvhTree {
    int BVHtree[];
};
layout(std430, binding = 12) buffer BVHleafTriIndices {
    int leafTriIndices[];
};
layout(std430, binding = 13) buffer objIndicesbuffer {
    int objIndices[];
};
layout(std430, binding = 14) buffer mtlBuffer {
    float mtlData[];
};
layout(std430, binding = 15) readonly buffer textureHandles {
    sampler2D textures[];
};

vec3 sampleTexture(int textureIndex, vec2 uv) {
    return texture(textures[textureIndex], uv).rgb;
}

struct raySceneResult {
    vec3 loc;
    vec3 dir;
    vec3 norm;
    vec3 tangent;
    int material;
    int mtl;
    vec2 uvSample;
    vec3 col;
    int type;
    int id;
    float distance;
    int parentID;
};
struct mtl {
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    float Ns;// specular exponent
    float d;// dissolved (transparency 1-0, 1 is opaque)
    float Tr;// occasionally used, opposite of d (0 is opaque)
    vec3 Tf;// transmission filter
    float Ni;// refractive index
    vec3 Ke;// emission color
    float Density;
    int illum;// shading model (0-10, each has diff properties)
    int map_Ka;
    int map_Kd;
    int map_Ks;
//PBR extension types
    float Pm;// metallicity (0-1, dielectric to metallic)
    float Pr;// roughness (0-1, perfectly smooth to "extremely" rough)
    float Ps;// sheen (0-1, no sheen effect to maximum sheen)
    float Pc;// clearcoat thickness (0-1, smooth clearcoat to rough clearcoat (blurry reflections))
    float Pcr;
    float aniso;// anisotropy (0-1, isotropic surface to fully anisotropic) (uniform-directional reflections)
    float anisor;// rotational anisotropy (0-1, but essentially 0-2pi, rotates pattern of anisotropy)
    int map_Pm;
    int map_Pr;
    int map_Ps;
    int map_Pc;
    int map_Pcr;
    int map_norm;
    int map_d;
    int map_Tr;
    int map_Ns;
    int map_Ke;
//CUSTOM
    float subsurface;
    vec3 subsurfaceColor;
    vec3 subsurfaceRadius;
};

const int MAX_REFRACTIONSTACK = 10;
float refractionIndiceStack[MAX_REFRACTIONSTACK];
int refractionIndiceStackSize = 0;
void clearIndiceStack() {
    refractionIndiceStackSize = 0;
}
void addToIndiceStack(float newElement) {
    if (refractionIndiceStackSize < MAX_REFRACTIONSTACK) {
        for (int i = refractionIndiceStackSize; i > 0; i--) {
            refractionIndiceStack[i] = refractionIndiceStack[i - 1];
        }
        refractionIndiceStack[0] = newElement;
        refractionIndiceStackSize++;
    }
}
void removeFirstOfIndiceStack() {
    if (refractionIndiceStackSize > 0) {
        for (int i = 0; i < refractionIndiceStackSize - 1; i++) {
            refractionIndiceStack[i] = refractionIndiceStack[i + 1];
        }
        refractionIndiceStackSize--;
    }
}

bool RAY_IN_OBJECT;
bool RANDOM_WALKING;
bool APPLY_ABSORBTION;
bool APPLY_SUBSURFACE;
vec3 ABSORB = vec3(-1);
vec3 RAY_ENTER_LOCATION = vec3(0);
float DISTANCE_TRAVELED = 0;

int me = int(mtlData[0]);

mtl newMtl(int m) {
    mtl Out;
    Out.Ka = vec3(mtlData[me*m + 1], mtlData[me*m + 2], mtlData[me*m + 3]);
    Out.Kd = vec3(mtlData[me*m + 4], mtlData[me*m + 5], mtlData[me*m + 6]);
    Out.Ks = vec3(mtlData[me*m + 7], mtlData[me*m + 8], mtlData[me*m + 9]);
    Out.Ns = mtlData[me*m + 10];
    Out.d = mtlData[me*m + 11];
    Out.Tr = mtlData[me*m + 12];
    Out.Tf = vec3(mtlData[me*m + 13], mtlData[me*m + 14], mtlData[me*m + 15]);
    Out.Ni = mtlData[me*m + 16];
    Out.Ke = vec3(mtlData[me*m + 17], mtlData[me*m + 18], mtlData[me*m + 19]);
    Out.Density = mtlData[me*m + 20];
    Out.illum = int(mtlData[me*m + 21]);
    Out.map_Ka = int(mtlData[me*m + 22]);
    Out.map_Kd = int(mtlData[me*m + 23]);
    Out.map_Ks = int(mtlData[me*m + 24]);
    //pbr extension
    Out.Pm = mtlData[me*m + 25];
    Out.Pr = mtlData[me*m + 26];
    Out.Ps = mtlData[me*m + 27];
    Out.Pc = mtlData[me*m + 28];
    Out.Pcr = mtlData[me*m + 29];
    Out.aniso = mtlData[me*m + 30];
    Out.anisor = mtlData[me*m + 31];
    Out.map_Pm = int(mtlData[me*m + 32]);
    Out.map_Pr = int(mtlData[me*m + 33]);
    Out.map_Ps = int(mtlData[me*m + 34]);
    Out.map_Pc = int(mtlData[me*m + 35]);
    Out.map_Pcr = int(mtlData[me*m + 36]);
    Out.map_norm = int(mtlData[me*m + 37]);
    Out.map_d = int(mtlData[me*m + 38]);
    Out.map_Tr = int(mtlData[me*m + 39]);
    Out.map_Ns = int(mtlData[me*m + 40]);
    Out.map_Ke = int(mtlData[me*m + 41]);
    //CUSTOM
    Out.subsurface = mtlData[me*m + 42];
    Out.subsurfaceColor = vec3(mtlData[me*m + 43], mtlData[me*m + 44], mtlData[me*m + 45]);
    Out.subsurfaceRadius = vec3(mtlData[me*m + 46], mtlData[me*m + 47], mtlData[me*m + 48]);
    return Out;
}
mtl mapMtl(mtl M, vec2 uv) {
    // Mapped materials are reset to mapped values, otherwise are unchanged
    mtl m = M;
    m.Ka = ((M.map_Ka > -1) ? sampleTexture(M.map_Ka, uv) * M.Ka: M.Ka);
    m.Kd = ((M.map_Kd > -1) ? sampleTexture(M.map_Kd, uv) * M.Kd : M.Kd);
    m.Ks = ((M.map_Ks > -1) ? sampleTexture(M.map_Ks, uv) : M.Ks);
    m.Ke = ((M.map_Ke > -1) ? sampleTexture(M.map_Ke, uv) : M.Ke);
    m.d = ((M.map_d > -1) ? sampleTexture(M.map_d, uv).r : M.d);
    m.Tr = ((M.map_Tr > -1) ? sampleTexture(M.map_Tr, uv).r : M.Tr);
    m.Ns = ((M.map_Ns > -1) ? sampleTexture(M.map_Ns, uv).r : M.Ns);
    m.Pm = ((M.map_Pm > -1) ? sampleTexture(M.map_Pm, uv).r : M.Pm);
    m.Pr = ((M.map_Pr > -1) ? sampleTexture(M.map_Pr, uv).r : M.Pr);
    m.Ps = ((M.map_Ps > -1) ? sampleTexture(M.map_Ps, uv).r : M.Ps);
    m.Pc = ((M.map_Pc > -1) ? sampleTexture(M.map_Pc, uv).r : M.Pc);
    return m;
}

const float NO_HIT = sqrt(-1);
const float EPSILON = 1e-10;
const float Gr = 0.5 + sqrt(5) / 2;
const float Gr2 = pow(Gr, 2);
int numObj = objIndices[0];
int numImplicits = int(ImpData[0]);
int numEllipsoids = int(EllipData[0]);

vec3 bgCol(vec3 In) {
    vec2 uv = vec2(
    0.5 + atan(In.z, In.x) / (2.0 * 3.14159),
    0.5 - asin(In.y) / 3.14159
    );
    vec3 skyColor = sampleTexture(0, uv);
    return skyColor.rgb;
}

mat3 rotateX(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
    1.0, 0.0, 0.0,
    0.0,   c,  -s,
    0.0,   s,   c
    );
}
mat3 rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
    c, 0.0,   s,
    0.0, 1.0, 0.0,
    -s, 0.0,   c
    );
}
mat3 rotateZ(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
    c,  -s, 0.0,
    s,   c, 0.0,
    0.0, 0.0, 1.0
    );
}
mat3 rotationMatrix(vec3 angles) {
    return rotateX(angles.x) * rotateY(angles.y) * (angles.z != 0 ? rotateZ(angles.z) : mat3(1));
}
vec3 rotate(vec3 p, vec3 rot) {
    float cx = cos(rot.x);
    float sx = sin(rot.x);
    float cy = cos(rot.y);
    float sy = sin(rot.y);
    float cz = cos(rot.z);
    float sz = sin(rot.z);
    mat3 rm = rotationMatrix(rot);
    return p * rm;
}
vec3 rotateBack(vec3 p, vec3 rot) {
    float cx = cos(rot.x);
    float sx = sin(rot.x);
    float cy = cos(rot.y);
    float sy = sin(rot.y);
    float cz = cos(rot.z);
    float sz = sin(rot.z);
    mat3 rm = mat3(
    cy * cz, cx * sz + cz * sx * sy, sx * sz - cx * cz * sy,
    -cy * sz, cx * cz - sx * sy * sz, cz * sx + cx * sy * sz,
    sy, -cy * sx, cx * cy
    );
    return p * rm;
}

float funcs(float x, float y, float z, int fn) {
    if (fn == 1) {
        return x*x + y*y + z*z - 15;
    }
    if (fn == 2) {
        return sin(x / 2) + sin(z / 2) + y / 2 + 10;
    }
    if (fn == 3) {
        return 2*x*x + y*y + 2*z*z - 3;
    }
    if (fn == 4) {
        //tractor beam
        float fx = (y < -0.63) ? 0.2 * (y - 5) : sqrt(-1.0);
        return x*x + z*z - fx*fx;
    }
    if (fn == 5) {
        float X = x;
        float Y = y;
        float Z = z;
        float Y2 = pow(Y, 2);
        float X2 = pow(X, 2);
        float Z2 = pow(Z, 2);
        float w = 1;
        return 4 * (Gr2 * X2 - Y2) * (Gr2 * Y2 - Z2) * (Gr2 * Z2 - X2) - (1 + 2 * Gr) * pow((X2 + Y2 + Z2 - w), 2) * w;
    }
    if (fn == 6) {
        //ufo
        float fx = y > -0.7417 ? (y > 0 ? (y < 1.3 ? pow(y, 3) - 3 : -sqrt(-1)) : (y > -2 ? 10*sin(y+4.7)+7 : sqrt(-1))) : sqrt(-1);
        return x*x + z*z - fx*fx;
    }
    if (fn == 7) {
        //ufo top
        float fx = -1.047 * sqrt(-(y-1.25)*(y-1.25) + 1);
        return x*x + z*z - fx*fx;
    }
    if (fn == 8) {
        //
        float fx = (y > 7.6 ? sqrt(-1) : (y < 0 ? sqrt(-1) : (y < .196 ? 0.5625 * cos(17 * y) + 0.6875 : 0.125 + pow(sin(0.9 * pow(y / 2.5, 0.6)), 10))));
        return x * x + z * z - fx * fx;
    }
    if (fn == 9) {
        float Y = -y;
        float fx = (Y < 2 ? NO_HIT : (Y > 8 ? NO_HIT : (Y > 7.0 ? -pow(Y - 7.0, 10.0) + 1.0 : (Y > 5.0 ? 1.0 : (Y > 4.1 ? sin(Y - 0.27) : (Y > 2.5 ? (Y + 0.9826) / 8.0 : (Y > 2.0 ? sin(40.0 * Y) / 40.0 + 0.45 : 0.0)))))));
        return x * x + z * z - fx * fx;
    }
    if (fn == 10) {
        float fx = y > 0.0 ? (y < 5.488 ? -pow(10.0, -5.0 * y) + 1.04 : (y < 6.74 ? 0.34 * sin(2.5 * y + 6.7) + 0.7 : (y < 8.43 ? 0.36 : (y < 8.76 ? -50.0 * (y - 8.6) * (y - 8.6) * (y - 8.6) * (y - 8.6) + 0.4 : (y < 9.0 ? 0.36 : NO_HIT))))) : NO_HIT;
        return x * x + z * z - fx * fx;
    }
    return 1e30;
}

vec3 rayTri(vec3 o, vec3 d, vec3 v1, vec3 v2, vec3 v3) {
    vec3 e1 = v2 - v1;
    vec3 e2 = v3 - v1;
    vec3 dCross_e2 = cross(d, e2);
    float det = dot(e1, dCross_e2);
    if (abs(det) < EPSILON) {
        return vec3(1e30);
    }
    float invDet = 1.0 / det;
    vec3 s = o - v1;
    float u = dot(s, dCross_e2) * invDet;
    if (u < 0.0 || u > 1.0) {
        return vec3(1e30);
    }
    vec3 sCross_e1 = cross(s, e1);
    float v = dot(d, sCross_e1) * invDet;
    if (v < 0.0 || u + v > 1.0) {
        return vec3(1e30);
    }
    float t = dot(e2, sCross_e1) * invDet;
    return (t > EPSILON) ? vec3(t, u, v) : vec3(1e30);
}
float rayEllipsoid(vec3 o, vec3 d, vec3 c, float r, float f, float g, float h) {
    float a = f*d.x*d.x + g*d.y*d.y + h*d.z*d.z;
    float b = 2*(f*((o-c).x)*d.x + g*((o-c).y)*d.y + h*((o-c).z)*d.z);
    float C = f*(o-c).x*(o-c).x + g*(o-c).y*(o-c).y + h*(o-c).z*(o-c).z - r*r;
    float Discriminant = b*b - 4*a*C;
    float t = (sqrt(Discriminant)-b)/(2*a);
    float tAlt = (-b-sqrt(Discriminant))/(2*a);
    if (Discriminant > 0 && (tAlt > 0) || (t > 0)) {
        return (t > tAlt ? tAlt : t);
    }
    return 1e30;
}
float rayImplicit(vec3 o, vec3 d, int fn) {
    return 1e30;
    /*
    for (float i = 0; i < 100; i += 0.1) {
        vec3 R = o + d * i;
        vec3 Rn = o + (d*(i+0.1));
        float fR = funcs(R.x, R.y, R.z, fn);
        float fRn = funcs(Rn.x, Rn.y, Rn.z, fn);
        if (abs(fR/(fR-fRn)) <= 20) {
            for (float j = i; j < 0.1+i; j += 0.001) {
                vec3 R = o + d * j;
                vec3 Rn = o + (normalize(d)*(j+0.001));
                float fR = funcs(R.x, R.y, R.z, fn);
                float fRn = funcs(Rn.x, Rn.y, Rn.z, fn);
                if (abs(fR/(fR-fRn)) <= 1) {
                    return j;
                }
            }
        }
    }
    return 1e30;
    */
}
float rayBox(vec3 o, vec3 d, vec3 Min, vec3 Max) {
    vec3 invD = 1.0/d;
    vec3 tMin = (Min - o) * invD;
    vec3 tMax = (Max - o) * invD;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);

    float dst = (tFar >= tNear && tFar > 0) ? tNear > 0 ? tNear : 0 : 1e30;
    return dst;
}

vec3 computeTangent(vec3 pos1, vec3 pos2, vec3 pos3, vec2 uv1, vec2 uv2, vec2 uv3, vec3 normal) {
    // Compute edge vectors
    vec3 edge1 = pos2 - pos1;
    vec3 edge2 = pos3 - pos1;

    // Compute UV delta
    vec2 deltaUV1 = uv2 - uv1;
    vec2 deltaUV2 = uv3 - uv1;

    // Compute tangent
    float f = 1.0 / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);

    vec3 tangent = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);

    // Orthogonalize tangent with respect to normal
    tangent = normalize(tangent - normal * dot(normal, tangent));

    return normalize(tangent);
}
vec3 gradient(vec3 p, int fn) {
    float partialDerivA = (funcs(p.x + 0.0001, p.y, p.z, fn) - funcs(p.x, p.y, p.z, fn)) / 0.0001;
    float partialDerivB = (funcs(p.x, p.y + 0.0001, p.z, fn) - funcs(p.x, p.y, p.z, fn)) / 0.0001;
    float partialDerivC = (funcs(p.x, p.y, p.z + 0.0001, fn) - funcs(p.x, p.y, p.z, fn)) / 0.0001;
    return vec3(-partialDerivA, -partialDerivB, -partialDerivC);
}

float rayNode(vec3 o, vec3 d, int node) {

    return rayBox(o,d,vec3(BVHdata[8 * node], BVHdata[8 * node + 1], BVHdata[8 * node + 2]),
                      vec3(BVHdata[8 * node + 3], BVHdata[8 * node + 4], BVHdata[8 * node + 5]));
}
raySceneResult rayBVH(vec3 o, vec3 d, int topLvlBVH, float previous_closest_t) {
    vec3 outputColor = vec3(0.0);
    int boxTests = 0;
    int triTests = 0;

    raySceneResult result;
    result.loc = vec3(1e30);
    result.type = -1;
    float closest_t = previous_closest_t;

    int INITstartIdx = int(BVHdata[8*topLvlBVH+6]);
    int INITendIdx = int(BVHdata[8*topLvlBVH+7]);

    int stack[64];
    int stackPtr = 0;

    if (rayNode(o,d,topLvlBVH) > closest_t) return result;

    stack[stackPtr++] = topLvlBVH;

    while (stackPtr > 0) {
        int currentNodeID = stack[--stackPtr];
        int leftChildID = BVHtree[3 * currentNodeID + 1];
        int rightChildID = BVHtree[3 * currentNodeID + 2];


        bool isLeaf = (leftChildID | rightChildID) == -1;
        if (isLeaf) {
            outputColor += vec3(0.1,0,0);
            int startIdx = int(BVHdata[8*currentNodeID+6]);
            int endIdx = int(BVHdata[8*currentNodeID+7]);
            for (int i = startIdx; i < endIdx; i++) {
                int currTriID = leafTriIndices[i];
                vec3 v1 = Triangles[currTriID].v1;
                vec3 v2 = Triangles[currTriID].v2;
                vec3 v3 = Triangles[currTriID].v3;
                vec3 hit = rayTri(o, d, v1, v2, v3);
                if (hit.x > 0.0 && hit.x < closest_t) {
                    closest_t = hit.x;
                    result.mtl = int(Triangles[currTriID].material.x);
                    mtl hitMtl = newMtl(result.mtl);
                    result.loc = hit;
                    result.dir = d;
                    result.id = currTriID;
                    int triIDreal = currTriID;
                    float u = hit.y;
                    float v = hit.z;
                    vec3 n1 = Triangles[currTriID].n1;

                    if (n1.x != 0 && n1.y != 0 && n1.z != 0) {
                        vec3 n2 = Triangles[currTriID].n2;
                        vec3 n3 = Triangles[currTriID].n2;
                        result.norm = normalize(n2*u + n3*v + (1.0-u-v)*n1);
                    } else {
                        result.norm = Triangles[currTriID].n2;
                    }
                    vec2 vt1 = Triangles[currTriID].vt1.xy;
                    if (vt1.x != 69.420) {
                        vec2 vt2 = Triangles[currTriID].vt2.xy;
                        vec2 vt3 = Triangles[currTriID].vt3.xy;
                        result.uvSample = vt2*u + vt3*v + (1-u-v)*vt1;
                        result.uvSample.y = 1-result.uvSample.y;
                        result.tangent = computeTangent(v1, v2, v3, vt1, vt2, vt3, result.norm);
                    } else {
                        result.uvSample = vec2(-1);
                    }
                    result.type = 1;
                }
            }
        } else {
            boxTests+=2;
            float Ldist = rayNode(o, d, max(0, leftChildID));
            float Rdist = rayNode(o, d, max(0, rightChildID));
            if (Ldist > Rdist) {
                if (Ldist < closest_t) stack[stackPtr++] = leftChildID;
                if (Rdist < closest_t) stack[stackPtr++] = rightChildID;
            } else {
                if (Rdist < closest_t) stack[stackPtr++] = rightChildID;
                if (Ldist < closest_t) stack[stackPtr++] = leftChildID;
            }
        }
    }
    result.col = outputColor*0.1 + vec3(0,0,exp(0.01*(boxTests-200))) + vec3(exp(0.02*(triTests-150)),0,0);
    //result.col = vec3(boxTests*0.01);
    return result;
}

vec3 debugRayScene(vec3 o, vec3 d) {
    vec3 ret = vec3(0);

    for (int I = 1; I < numObj+1; I++) {
        ret += rayBVH(o, d, objIndices[I], 1e30).col/(numObj);
    }

    return ret;
}
raySceneResult rayScene(vec3 o, vec3 d) {
    o = o+ 1e-4*d;
    vec3 N = vec3(0, 0, 0);
    vec3 hitTangent = vec3(0);
    float t = 1e30;
    float closest_t = 1e30;
    bool hit = false;
    // 0 = null, 1 = tri, 2 = implicit, 3 = ellipsoid
    int hitType = 0;
    int hitID = -1;
    int hitMat = -1;
    vec2 hitUV = vec2(0);
    int parentID = -1;
    raySceneResult result;

    for (int I = 1; I < numObj+1; I++) {
        int currObjID = objIndices[I];
        raySceneResult BVHresult = rayBVH(o, d, currObjID, closest_t);
        if (BVHresult.loc.x < closest_t) {
            closest_t = BVHresult.loc.x;
            N = BVHresult.norm;
            hitTangent = BVHresult.tangent;
            hitType = 1;
            hitMat = BVHresult.mtl;
            hitID = BVHresult.id;
            parentID = currObjID;
            hitUV = BVHresult.uvSample;
            hit = true;
        }
    }
    for (int i = 0; i < numImplicits; i++) {
        int fn = int(ImpData[i+1]);
        vec3 shift = vec3(ImpData[1+numImplicits+3*i], ImpData[1+numImplicits+3*i+1], ImpData[1+numImplicits+3*i+2]);
        vec3 scale = vec3(ImpData[1 + numImplicits + numImplicits*3 + 3*i], ImpData[1 + numImplicits + numImplicits*3 + 3*i + 1], ImpData[1 + numImplicits + numImplicits*3 + 3*i + 2]);
        vec3 rot = vec3(ImpData[1+ numImplicits + numImplicits*6 + 3*i], ImpData[1+ numImplicits + numImplicits*6 + 3*i + 1], ImpData[1+ numImplicits + numImplicits*6 + 3*i + 2]);
        int mat = int(ImpData[1 + numImplicits + numImplicits*9 + i]);
        vec3 O = (o-shift) / scale;
        vec3 D = d / scale;
        float t = 0;
        bool rotated = (length(rot)>0);
        if (rotated) {
            t = rayImplicit(rotate(O, rot), rotate(D, rot), fn);
        } else {
            t = rayImplicit(O, D, fn);
        }
        if (t < closest_t) {
            closest_t = t;
            hitMat = mat;
            if (rotated) {
                N = -normalize(rotateBack(gradient(rotate(O, rot) + t*rotate(D, rot), fn), rot));
            } else {
                N = -normalize(gradient(O+t*D, fn));
            }
            hit = true;
            hitType = 2;
            hitID = i;
        }
    }
    for (int i = 0; i < numEllipsoids; i++) {
        vec3 c = vec3(EllipData[1 + 3*i], EllipData[1 + 3*i + 1], EllipData[1 + 3*i + 2]);
        vec3 stretch = vec3(EllipData[1 + numEllipsoids*3 + 3*i], EllipData[1 + numEllipsoids*3 + 3*i + 1], EllipData[1 + numEllipsoids*3 + 3*i + 2]);
        vec3 rot = vec3(EllipData[1 + numEllipsoids*6 + 3*i], EllipData[1 + numEllipsoids*6 + 3*i + 1], EllipData[1 + numEllipsoids*6 + 3*i + 2]);
        float r = EllipData[1 + numEllipsoids*9 + i];
        int mat = int(EllipData[1 + numEllipsoids*10 + i]);
        float t = 0;
        bool rotated = (length(rot)>0);
        if (rotated) {
            t = rayEllipsoid(rotate(o, rot), rotate(d, rot), c, r, stretch.x, stretch.y, stretch.z);
        } else {
            t = rayEllipsoid(o, d, c, r, stretch.x, stretch.y, stretch.z);
        }
        if (t < closest_t) {
            closest_t = t;
            hitMat = mat;
            if (rotated) {
                N = normalize(rotateBack(o+t*d - c, rot));
            } else {
                N = normalize(o+t*d - c);
            }
            hit = true;
            hitType = 3;
            hitID = i;
        }
    }
    result.type = hitType;
    result.id = hitID;
    if (closest_t<1e25) {
        result.loc = o+closest_t*d;
        result.dir = normalize(d);
        result.norm = N;
        result.tangent = hitTangent;
        result.material = hitMat;
        result.uvSample = hitUV;
        result.distance = closest_t;
        result.parentID = parentID;
        return result;
    }
    result.loc = vec3(1e30, 1e30, 1e30);
    result.dir = d;
    result.norm = vec3(0, 0, 0);
    result.tangent = vec3(0);
    result.material = -1;
    result.uvSample = hitUV;
    result.distance = -1;
    return result;
}

vec3 directDiffuse(vec3 o, vec3 d) {
    raySceneResult hit = rayScene(o,d);
    if (hit.id > -1) {
        mtl m = newMtl(hit.material);
        m = mapMtl(m, hit.uvSample);
        vec3 N = (m.map_norm > -1 ? sampleTexture(m.map_norm, hit.uvSample) : hit.norm);
        vec3 col =  m.Ka + m.Kd*0.2 + (m.Kd*dot(vec3(0, 1, 0), N)) + m.Ke;
        if (m.subsurface > 0) {
            vec3 subsurfaceColor = m.subsurfaceColor;
            vec3 subsurfaceRadius = m.subsurfaceRadius;
            float density = m.Density;

            // Calculate distance through the object
            float si = distance(o, rayBVH(hit.loc, d, hit.parentID, 1e30).loc);

            // Compute attenuation using subsurface parameters
            vec3 sigma_t = 1.0 / max(subsurfaceRadius, vec3(1e-4));
            vec3 subsurfaceLight = exp(-sigma_t * si) * subsurfaceColor;

            // Add subsurface contribution to indirect light
            col = subsurfaceLight;
        }
        return col;
    } else {
        return bgCol(d);
    }
}

uint InitializeState(int seed) {
    return uint(seed);
}
uint NextRandom(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint result = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result;
}
float random(inout uint state) {
    return NextRandom(state) / 4294967295.0; // 2^32 - 1
}
// Random value in normal distribution (mean=0, sd=1)
float randValNormalDist(inout uint rngState) {
    // Thanks to https://stackoverflow.com/a/6178290
    float theta = 2.0 * 3.1415926 * random(rngState);
    float rho = sqrt(-2.0 * log(random(rngState)));
    return rho * cos(theta);
}
vec3 randLambertianDistVec(inout uint rngState) {
    float randx = randValNormalDist(rngState);
    float randy = randValNormalDist(rngState);
    float randz = randValNormalDist(rngState);
    vec3 randVec = vec3(randx,randy,randz);
    return randVec;
}
vec3 randUniformDistVec(inout uint rngState) {
    float randx = random(rngState) - 0.5;
    float randy = random(rngState) - 0.5;
    float randz = random(rngState) - 0.5;
    return vec3(randx, randy, randz) * 2;
}
vec2 randUniformDistDiskSample(inout uint rngState) {
    return normalize((vec2(random(rngState), random(rngState))-0.5) * 2);
}
vec3 randDiskSamplePoint(inout uint rngState, vec3 n) {
    vec3 uvec = cross(n,vec3(0,1,0));
    vec3 vvec = cross(n,uvec);
    vec2 sampleUV = randUniformDistDiskSample(rngState);
    return uvec*sampleUV.x + vvec*sampleUV.y;
}


float fresnelReflectAmount(float n1, float n2, vec3 normal, vec3 incidence, float initReflectAmount) {
    float r0 = (n1-n2) / (n1+n2);
    r0 *= r0;
    float cosX = -dot(normal, incidence);
    if (n1 > n2)
    {
        float n = n1/n2;
        float sinT2 = n*n*(1.0-cosX*cosX);
        //TIR
        if (sinT2 > 1.0)
        return 1.0;
        cosX = sqrt(1.0-sinT2);
    }
    float x = 1.0-cosX;
    float ret = r0+(1.0-r0)*x*x*x*x*x;

    return ret;
}

vec4 chooseRay(mtl m, float n1, float n2, vec3 N, vec3 D, inout uint rngState) {
    float reflectionWeight = (1-m.Pr);
    float clearcoatWeight = m.Pc;
    float transmissionWeight = (m.Tr > 0 ? m.Tr : (m.Tf.x > 0 ? (m.Tf.x + m.Tf.y + m.Tf.z)/3 : 0));
    float subsurfaceWeight = m.subsurface;

    float eta = n1/n2;
    float fresnel = 0;
    if (m.illum == 5 || m.illum == 7 || transmissionWeight > 0) {
        fresnel = fresnelReflectAmount(n1, n2, N, D, reflectionWeight);
        reflectionWeight += fresnel*m.Pr;
        transmissionWeight *= (1-fresnel);
    }

    float diffuseWeight = (1-m.Pm) * (1-transmissionWeight) * (1-fresnel);

    float totalWeight = diffuseWeight + reflectionWeight + clearcoatWeight + transmissionWeight; //+ subsurfaceWeight;
    diffuseWeight /= totalWeight;
    reflectionWeight /= totalWeight;
    clearcoatWeight /= totalWeight;
    transmissionWeight /= totalWeight;
    //subsurfaceWeight /= totalWeight;

    float roll = random(rngState);

    int winType = 0;
    vec3 outDir;
    if (roll < reflectionWeight) {
        //reflect won
        winType = 1;
        outDir = mix(reflect(D, N), normalize(randLambertianDistVec(rngState) + N), 0);
    } else if (roll < reflectionWeight + clearcoatWeight) {
        //clearcoat won
        winType = 2;
        outDir = mix(reflect(D, N), normalize(randLambertianDistVec(rngState) + N), m.Pcr);
    } else if (roll < reflectionWeight + clearcoatWeight + transmissionWeight) {
        //transmission won
        winType = 3;
        outDir = refract(D, N, eta);
    } /*else if (roll < reflectionWeight + clearcoatWeight + transmissionWeight + subsurfaceWeight) {
        subsurface won
        winType = 4;
        outDir = refract(D, N, eta);
    }*/
    else {
        //subsurface or diffuse won
        if (subsurfaceWeight > 0) {
            if (random(rngState) < subsurfaceWeight) {
                //subsurface won
                winType = 4;
                outDir = normalize(randLambertianDistVec(rngState) + N);
            } else {
                //diffuse won
                winType = 0;
                outDir = normalize(randLambertianDistVec(rngState) + N);
            }
        } else {
            //diffuse won
            winType = 0;
            outDir = normalize(randLambertianDistVec(rngState) + N);
        }
    }

    return vec4(outDir,winType);
}
vec3 trace(vec3 o, vec3 d, inout uint rngState) {
    vec3 O = o;
    vec3 D = d;
    vec3 col = vec3(1);
    vec3 incLight = vec3(0);
    clearIndiceStack();
    addToIndiceStack(1.0029);
    RAY_IN_OBJECT = false;
    int bounce = 0;

    while (bounce < MAX_BOUNCES) {
        bounce++;
        raySceneResult hit = rayScene(O,D);
        if (hit.id > -1) {
            O = hit.loc;
            mtl hitMtl = newMtl(hit.material);
            mtl m = mapMtl(hitMtl,hit.uvSample);
            vec3 N = (m.map_norm > -1 ? sampleTexture(m.map_norm, hit.uvSample) : hit.norm);
            vec3 emission = m.Ke;
            float ND = dot(N,D);
            N *= (ND > 0 ? -1: 1);
            float n1 = 1;
            float n2 = 1;
            if (ND < 0) {
                addToIndiceStack(m.Ni);
                n1 = refractionIndiceStack[1];
                n2 = refractionIndiceStack[0];
            } else {
                n1 = refractionIndiceStack[0];
                n2 = refractionIndiceStack[1];
                removeFirstOfIndiceStack();
            }
            bool isSpecular = false;
            vec4 weightedRayChoice = chooseRay(m, n1, n2, N, D, rngState);
            if (weightedRayChoice.w == 2) isSpecular = true;
            vec3 lastD = D;
            D = weightedRayChoice.xyz;
            if (weightedRayChoice.w == 3) {
                if (ND < 0) {
                    // Do refraction and potentially Beers law
                    if (RAY_IN_OBJECT) {
                        // Ray was already in a refractive object, and is entering another. Do Beers law
                        DISTANCE_TRAVELED = distance(RAY_ENTER_LOCATION, O);
                        APPLY_ABSORBTION = true;
                    }
                    RAY_IN_OBJECT = true;
                    RAY_ENTER_LOCATION = O;
                } else {
                    // Ray is leaving a refractive object. Do attenuation
                    RAY_IN_OBJECT = false;
                    DISTANCE_TRAVELED = distance(RAY_ENTER_LOCATION,O);
                    APPLY_ABSORBTION = true;
                }
            }

            incLight += emission * col;
            if (length(col)<0.1) return incLight;
            if (APPLY_ABSORBTION) {
                col *= exp(-m.Tf * DISTANCE_TRAVELED * m.Density);
                APPLY_ABSORBTION = false;
            } else if (weightedRayChoice.w == 4) {
//AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
            } else {
                col *= (isSpecular ? m.Ks : m.Kd);
            }
        } else {
            if (bounce == 0) return bgCol(D);
            incLight += bgCol(D)*col;
            break;
        }
    }
    return incLight;
}

void main() {
    ivec2 pixel_coords = ivec2(texCoord * vec2(resolution, resolution * screenHratio));
    uint index = uint(pixel_coords.y) * uint(resolution) + uint(pixel_coords.x);
    if (any(greaterThanEqual(pixel_coords, ivec2(resolution, int(resolution * screenHratio))))) return;
    if (abs(pixel_coords.x-MOUSE_POS.x) < resolution*0.005 && abs(pixel_coords.y-MOUSE_POS.y) < resolution*0.005) {
        vec2 mouseUV = vec2(MOUSE_POS.x/resolution,MOUSE_POS.y/(resolution*screenHratio))*2.0 - 1.0;
        raySceneResult mouse_scene = rayScene(ORIGIN, rotate(vec3(((mouseUV) * vec2(-1, screenHratio) * screenSize), focalLength), ROTATION));
        fragColor = vec4(mouse_scene.norm,0);
        return;
    }
    vec3 direction = rotate(vec3(((texCoord * 2.0 - 1.0) * vec2(-1.0, screenHratio) * screenSize), focalLength), ROTATION);
    vec3 col = vec3(0);
    uint rngState = index+u_seed;
    if (DEBUG == 0) {
        for (int rayID = 0; rayID < SAMPLE_RES; rayID ++) {
            vec3 origin_jittered = ORIGIN + rotate(randLambertianDistVec(rngState) * BLUR, ROTATION);
            float internal_focal_distance = FOCAL_DISTANCE;
            if (AUTO_FOCUS == 1) {
                float mid_to_scene = rayScene(ORIGIN, rotate(vec3(0,0,1),ROTATION)).distance;
                if (mid_to_scene > 0) {
                    internal_focal_distance = mid_to_scene;
                }
            }
            vec3 focal_point = ORIGIN + direction * internal_focal_distance;
            vec3 direction_adjusted = normalize(focal_point - origin_jittered);
            if (RAYTRACING == 1) {
                col += trace(origin_jittered, direction_adjusted, rngState);
            } else {
                col += directDiffuse(origin_jittered, direction_adjusted);
            }
        }
        col /= SAMPLE_RES;
    } else {
        col = debugRayScene(ORIGIN, direction);
    }
//    vec4 lastFrameColor = imageLoad(FRAME, pixel_coords);
//    float invFc = 1.0 / float(u_frameCount + 1);
//    vec4 averagedColor = lastFrameColor * float(u_frameCount) * invFc + vec4(col, 1.0) * invFc;
//    imageStore(FRAME, pixel_coords, averagedColor);
//    fragColor = averagedColor;
    float frameCount = float(u_frameCount);
    if (frameCount == 1) {
        imageStore(FRAME, pixel_coords, vec4(col, 1));
        fragColor = vec4(col, 1);
    } else {
        vec4 lastFrameColor = imageLoad(FRAME, pixel_coords);
        vec4 newTotal = lastFrameColor + vec4(col, 1);
        imageStore(FRAME, pixel_coords, newTotal);
        fragColor = newTotal/frameCount;
    }
}