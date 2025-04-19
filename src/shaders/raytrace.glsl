#version 460
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : require
#define PI 3.141592
layout(local_size_x = 17, local_size_y = 17) in;
layout(rgba32f, binding = 0) uniform image2D FRAME;
uniform int u_frameCount;
uniform int u_seed;

layout(std430, binding = 0) buffer RayOrigins {
    vec3 origin;
};
layout(std430, binding = 2) buffer RayDirections {
    vec3 directions[];
};
layout(std430, binding = 3) buffer TriangleData {
    float TriData[];
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
};
//
layout(std430, binding = 5) buffer ImplicitData {
    float ImpData[];
};
layout(std430, binding = 6) buffer materialData {
    float MatData[];
};
layout(std430, binding = 7) buffer ellipsoidData {
    float EllipData[];
};
layout(std430, binding = 8) buffer emissiveData {
    float EmissiveData[];
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
    int material;
    int mtl;
    vec2 uvSample;
    vec3 col;
    int type;
    int id;
};
struct mat {
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    float alpha;
    float refl;
    float trans;
    float refrIndex;
    float emission;
    float emissiveRadius;
    bool solid;
    float absorption;
    int textureID;
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
    float EmissionStrength;
    int illum;// shading model (0-10, each has diff properties)
    int map_Ka;
    int map_Kd;
    int map_Ks;
//PBR extension types
    float Pm;// metallicity (0-1, dielectric to metallic)
    float Pr;// roughness (0-1, perfectly smooth to "extremely" rough)
    float Ps;// sheen (0-1, no sheen effect to maximum sheen)
    float Pc;// clearcoat thickness (0-1, smooth clearcoat to rough clearcoat (blurry reflections))
    float aniso;// anisotropy (0-1, isotropic surface to fully anisotropic) (uniform-directional reflections)
    float anisor;// rotational anisotropy (0-1, but essentially 0-2pi, rotates pattern of anisotropy)
    int map_Pm;
    int map_Pr;
    int map_Ps;
    int map_Pc;
    int map_norm;
    int map_d;
    int map_Tr;
    int map_Ns;
};
float rand(vec3 In) {
    return fract(sin(47.57891*In.x+In.y+In.z*425)*47678.786234);
}
/*
vec3 randVec(vec3 In) {
    return vec3(rand(In.x+u_seed/33.211),fract(-In.y*In.z/u_seed*0.331149),u_seed*In.x-In.y/671.444,rand(In.x+u_seed/831.911),fract(In.x/u_seed*68.41149),u_seed*In.y-In.z/5151.424);
}
*/

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
bool RAY_WAS_IN_OBJECT;
vec3 MAT_FILTER ;
vec3 ABSORB = vec3(-1);
vec3 RAY_ENTER_LOCATION = vec3(0);

int me = int(mtlData[0]);

mat newMat(int m) {
    //Ka, Kd, Ks, alpha, reflectivity (0-1, %), transmission (0-1, %), refractive index, emission (0-n)
    mat Out;
    Out.Ka = vec3(MatData[18*m + 1], MatData[18*m + 2], MatData[18*m + 3]);
    Out.Kd = vec3(MatData[18*m + 4], MatData[18*m + 5], MatData[18*m + 6]);
    Out.Ks = vec3(MatData[18*m + 7], MatData[18*m + 8], MatData[18*m + 9]);
    Out.alpha = MatData[18*m + 10];
    Out.refl = MatData[18*m + 11];
    Out.trans = MatData[18*m + 12];
    Out.refrIndex = MatData[18*m + 13];
    Out.emission = MatData[18*m + 14];
    Out.emissiveRadius = MatData[18*m + 15];
    Out.solid = (MatData[18*m + 16] == 0 ? false : true);
    Out.absorption = MatData[18*m + 17];
    Out.textureID = int(MatData[17*m + 18]);
    return Out;
}
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
    Out.EmissionStrength = mtlData[me*m + 20];
    Out.illum = int(mtlData[me*m + 21]);
    Out.map_Ka = int(mtlData[me*m + 22]);
    Out.map_Kd = int(mtlData[me*m + 23]);
    Out.map_Ks = int(mtlData[me*m + 24]);
    //pbr extension
    Out.Pm = mtlData[me*m + 25];
    Out.Pr = mtlData[me*m + 26];
    Out.Ps = mtlData[me*m + 27];
    Out.Pc = mtlData[me*m + 28];
    Out.aniso = mtlData[me*m + 29];
    Out.anisor = mtlData[me*m + 30];
    Out.map_Pm = int(mtlData[me*m + 31]);
    Out.map_Pr = int(mtlData[me*m + 32]);
    Out.map_Ps = int(mtlData[me*m + 33]);
    Out.map_Pc = int(mtlData[me*m + 34]);
    Out.map_norm = int(mtlData[me*m + 35]);
    Out.map_d = int(mtlData[me*m + 36]);
    Out.map_Tr = int(mtlData[me*m + 37]);
    Out.map_Ns = int(mtlData[me*m + 38]);
    return Out;
}

const float NO_HIT = sqrt(-1);
const float EPSILON = 1e-10;
const float Gr = 0.5 + sqrt(5) / 2;
const float Gr2 = pow(Gr, 2);
int numTri = int(TriData[0]);
int numVert = int(TriData[1]);
int numNorm = int(TriData[2]);
int numTexCoords = int(TriData[3]);
int numEmissives = int(EmissiveData[0]);
int numObj = objIndices[0];
int v_offset = 3 * numTri + 3;
int n_offset = v_offset + 3*numVert + 3*numTri;
int numImplicits = int(ImpData[0]);
int numEllipsoids = int(EllipData[0]);
int numMats = int(MatData[0]);

vec3 bgCol(vec3 In) {
    vec3 dir = normalize(In);

    vec2 uv = vec2(
    0.5 + atan(dir.z, dir.x) / (2.0 * 3.14159), // U coordinate
    0.5 - asin(dir.y) / 3.14159// V coordinate
    );


    vec3 skyColor = sampleTexture(0, uv);
    return skyColor.rgb;
}

vec3 rotate(vec3 p, vec3 rot) {
    float cx = cos(rot.x);
    float sx = sin(rot.x);
    float cy = cos(rot.y);
    float sy = sin(rot.y);
    float cz = cos(rot.z);
    float sz = sin(rot.z);
    mat3 rm = mat3(
    cy * cz, -cy * sz, sy,
    cx * sz + cz * sx * sy, cx * cz - sx * sy * sz, -cy * sx,
    sx * sz - cx * cz * sy, cz * sx + cx * sy * sz, cx * cy
    );
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
    vec3 e1 = v2-v1;
    vec3 e2 = v3-v1;
    vec3 dCross_e2 = cross(d, e2);
    float det = dot(e1, dCross_e2);
    if (det > -EPSILON && det < EPSILON) {
        return vec3(1e30);
    }
    float invDet = 1/det;
    vec3 s = o-v1;
    float u = invDet*dot(s, dCross_e2);
    if (u < 0 || u > 1) {
        return vec3(1e30);
    }
    vec3 sCross_e1 = cross(s, e1);
    float v = invDet*dot(d, sCross_e1);
    if (v < 0 || u+v > 1) {
        return vec3(1e30);
    }
    float t = invDet*dot(e2, sCross_e1);
    if (t > EPSILON) {
        return vec3(t, u, v);
    }
    return vec3(1e30);
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

vec3 gradient(vec3 p, int fn) {
    float partialDerivA = (funcs(p.x + 0.0001, p.y, p.z, fn) - funcs(p.x, p.y, p.z, fn)) / 0.0001;
    float partialDerivB = (funcs(p.x, p.y + 0.0001, p.z, fn) - funcs(p.x, p.y, p.z, fn)) / 0.0001;
    float partialDerivC = (funcs(p.x, p.y, p.z + 0.0001, fn) - funcs(p.x, p.y, p.z, fn)) / 0.0001;
    return vec3(-partialDerivA, -partialDerivB, -partialDerivC);
}

bool rayBox(vec3 o, vec3 d, vec3 Min, vec3 Max) {
    vec3 t_enter = vec3(((d.x>=0?Min.x:Max.x)-o.x)/d.x, ((d.y>=0?Min.y:Max.y)-o.y)/d.y, ((d.z>=0?Min.z:Max.z)-o.z)/d.z);
    vec3 t_exit = vec3(((d.x>=0?Max.x:Min.x)-o.x)/d.x, ((d.y>=0?Max.y:Min.y)-o.y)/d.y, ((d.z>=0?Max.z:Min.z)-o.z)/d.z);
    float te = max(max(t_enter.x, t_enter.y), t_enter.z);
    float tE = min(min(t_exit.x, t_exit.y), t_exit.z);
    return (te <= tE && tE >= 0.0);
}

raySceneResult rayBVH(vec3 o, vec3 d, int topLvlBVH) {
    const int MAX_STACK = 64;
    int stack[MAX_STACK];
    int stackTop = 0;
    raySceneResult result;
    result.loc = vec3(1e30);
    result.type = -1;
    float closest_t = 1e30;
    stack[0] = topLvlBVH;
    vec3 min = vec3(BVHdata[8*topLvlBVH], BVHdata[8*topLvlBVH+1], BVHdata[8*topLvlBVH+2]);
    vec3 max = vec3(BVHdata[8*topLvlBVH+3], BVHdata[8*topLvlBVH+4], BVHdata[8*topLvlBVH+5]);
    if (!rayBox(o, d, min, max)) {
        return result;
    }
    while (stackTop >= 0) {
        int currentID = stack[stackTop--];
        vec3 min = vec3(BVHdata[8*currentID], BVHdata[8*currentID+1], BVHdata[8*currentID+2]);
        vec3 max = vec3(BVHdata[8*currentID+3], BVHdata[8*currentID+4], BVHdata[8*currentID+5]);
        if (!rayBox(o, d, min, max)) {
            continue;
        }
        int index = 3*currentID;
        int leftChildID = BVHtree[index+1];
        int rightChildID = BVHtree[index+2];
        if (leftChildID == -1 && rightChildID == -1) {
            int startIdx = int(BVHdata[8*currentID+6]);
            int endIdx = int(BVHdata[8*currentID+7]);
            for (int i = startIdx; i < endIdx; i++) {
                int currTriID = leafTriIndices[i];
                int ft_1 = int(3 * (TriData[4 + 3*(currTriID)]-1));
                int ft_2 = int(3 * (TriData[4 + 1 + 3*(currTriID)]-1));
                int ft_3 = int(3 * (TriData[4 + 2 + 3*(currTriID)]-1));
                vec3 v1 = vec3(TriData[v_offset+ft_1+1], TriData[v_offset+ft_1+2], TriData[v_offset+ft_1+3]);
                vec3 v2 = vec3(TriData[v_offset+ft_2+1], TriData[v_offset+ft_2+2], TriData[v_offset+ft_2+3]);
                vec3 v3 = vec3(TriData[v_offset+ft_3+1], TriData[v_offset+ft_3+2], TriData[v_offset+ft_3+3]);
                vec3 hit = rayTri(o, normalize(d), v1, v2, v3);
                if (hit.x < closest_t && hit.x > 0.0) {
                    closest_t = hit.x;
                    result.mtl = int(TriData[6*numTri + 3*numVert + 3*numNorm + currTriID + 4]);
                    result.loc = hit;
                    result.dir = d;
                    result.id = currTriID;
                    int triIDreal = currTriID;
                    n_offset = 4 + 3*numVert + 6*numTri;
                    float u = hit.y;
                    float v = hit.z;
                    int fn_1 = 3*(int(TriData[4 + 3*numTri + 3*numVert + 3*(triIDreal)]) - 1);
                    if (fn_1 > -1) {
                        int fn_2 = 3*(int(TriData[4 + 3*numTri + 3*numVert + 3*(triIDreal) + 1]) - 1);
                        int fn_3 = 3*(int(TriData[4 + 3*numTri + 3*numVert + 3*(triIDreal) + 2]) - 1);
                        vec3 n1 = vec3(TriData[n_offset + fn_1], TriData[n_offset + fn_1 + 1], TriData[n_offset + fn_1 + 2]);
                        vec3 n2 = vec3(TriData[n_offset + fn_2], TriData[n_offset + fn_2 + 1], TriData[n_offset + fn_2 + 2]);
                        vec3 n3 = vec3(TriData[n_offset + fn_3], TriData[n_offset + fn_3 + 1], TriData[n_offset + fn_3 + 2]);
                        result.norm = normalize(n2*u + n3*v + (1.0-u-v)*n1);
                    } else {
                        result.norm = normalize(cross(v2-v1, v3-v1));
                    }
                    int triTexCoordIndicesStart = 4 + (numTri * 3) + (numVert * 3) + (numTri * 3) + (numNorm * 3) + numTri;
                    int texCoordsStart = triTexCoordIndicesStart + (numTri * 3);
                    int ftv_1 = int(TriData[triTexCoordIndicesStart + (currTriID * 3) + 0] - 1);
                    if (ftv_1 > -1) {
                        int ftv_2 = int(TriData[triTexCoordIndicesStart + (currTriID * 3) + 1] - 1);
                        int ftv_3 = int(TriData[triTexCoordIndicesStart + (currTriID * 3) + 2] - 1);
                        vec2 vt1 = vec2(TriData[texCoordsStart + (ftv_1 * 2) + 0], TriData[texCoordsStart + (ftv_1 * 2) + 1]);
                        vec2 vt2 = vec2(TriData[texCoordsStart + (ftv_2 * 2) + 0], TriData[texCoordsStart + (ftv_2 * 2) + 1]);
                        vec2 vt3 = vec2(TriData[texCoordsStart + (ftv_3 * 2) + 0], TriData[texCoordsStart + (ftv_3 * 2) + 1]);
                        result.uvSample = vt2*u + vt3*v + (1-u-v)*vt1;
                        result.uvSample.y = 1-result.uvSample.y;
                    } else {
                        result.uvSample = vec2(-1);
                    }
                    result.type = 1;
                }
            }
        } else {
            // Internal node - push children to stack
            // Push farther child first so closer child is processed first
            float t_left = 1e30;
            float t_right = 1e30;

            if (leftChildID != -1) {
                vec3 left_min = vec3(BVHdata[8*leftChildID], BVHdata[8*leftChildID+1], BVHdata[8*leftChildID+2]);
                vec3 left_max = vec3(BVHdata[8*leftChildID+3], BVHdata[8*leftChildID+4], BVHdata[8*leftChildID+5]);
                if (rayBox(o, d, left_min, left_max)) {
                    t_left = distance(o, (left_min + left_max) * 0.5);
                }
            }

            if (rightChildID != -1) {
                vec3 right_min = vec3(BVHdata[8*rightChildID], BVHdata[8*rightChildID+1], BVHdata[8*rightChildID+2]);
                vec3 right_max = vec3(BVHdata[8*rightChildID+3], BVHdata[8*rightChildID+4], BVHdata[8*rightChildID+5]);
                if (rayBox(o, d, right_min, right_max)) {
                    t_right = distance(o, (right_min + right_max) * 0.5);
                }
            }

            // Push in far-to-near order
            if (t_left > t_right) {
                if (leftChildID != -1 && t_left != 1e30) stack[++stackTop] = leftChildID;
                if (rightChildID != -1 && t_right != 1e30) stack[++stackTop] = rightChildID;
            } else {
                if (rightChildID != -1 && t_right != 1e30) stack[++stackTop] = rightChildID;
                if (leftChildID != -1 && t_left != 1e30) stack[++stackTop] = leftChildID;
            }
        }
    }

    return result;
}

raySceneResult rayScene(vec3 o, vec3 d) {
    vec3 N = vec3(0, 0, 0);
    float t = 1e30;
    float closest_t = 1e30;
    bool hit = false;
    // 0 = null, 1 = tri, 2 = implicit, 3 = ellipsoid
    int hitType = 0;
    int hitID = -1;
    int hitMat = -1;
    vec2 hitUV = vec2(0);
    raySceneResult result;

    for (int I = 1; I < numObj+1; I++) {
        raySceneResult BVHresult = rayBVH(o, d, objIndices[I]);
        if (BVHresult.loc.x < closest_t) {
            closest_t = BVHresult.loc.x;
            N = BVHresult.norm;
            hitType = 1;
            hitMat = BVHresult.mtl;
            hitID = BVHresult.id;
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
        result.material = hitMat;
        result.uvSample = hitUV;
        return result;
    }
    result.loc = vec3(1e30, 1e30, 1e30);
    result.dir = d;
    result.norm = vec3(0, 0, 0);
    result.material = -1;
    result.col = bgCol(d);
    result.uvSample = hitUV;
    return result;
}

raySceneResult rayObject(vec3 o, vec3 d, int type, int id) {
    raySceneResult Out;
    return Out;
}

mtl mapMtl(mtl M, vec2 uv) {
    mtl m;
    m.Ka = ((M.map_Ka > -1) ? sampleTexture(M.map_Ka, uv) * M.Ka: M.Ka);
    m.Kd = ((M.map_Kd > -1) ? sampleTexture(M.map_Kd, uv) : M.Kd);
    m.Ks = ((M.map_Ks > -1) ? sampleTexture(M.map_Ks, uv) : M.Ks);
    m.aniso = M.aniso;
    m.anisor = M.anisor;
    m.d = ((M.map_d > -1) ? sampleTexture(M.map_d, uv).r : M.d);
    m.Tr = ((M.map_Tr > -1) ? sampleTexture(M.map_Tr, uv).r : M.Tr);
    m.EmissionStrength = M.EmissionStrength;
    m.illum = M.illum;
    m.Ni = M.Ni;
    m.Ns = ((M.map_Ns > -1) ? sampleTexture(M.map_Ns, uv).r : M.Ns);
    m.Tf = M.Tf;
    m.Pm = ((M.map_Pm > -1) ? sampleTexture(M.map_Pm, uv).r : M.Pm);
    m.Pr = ((M.map_Pr > -1) ? sampleTexture(M.map_Pr, uv).r : M.Pr);
    m.Ps = ((M.map_Ps > -1) ? sampleTexture(M.map_Ps, uv).r : M.Ps);
    m.Pc = ((M.map_Pc > -1) ? sampleTexture(M.map_Pc, uv).r : M.Pc);
    m.map_norm = M.map_norm;
    return m;
}

float randValNormalDist(float In) {
    float theta = 2*PI*In;
    float rho = sqrt(-2 * log(rand(vec3(In,In*31.44,In-64.1676))));
    return rho*cos(theta);
}

vec3 hemisphereSample(vec3 normal, vec2 seed) {
    float randx = randValNormalDist(rand(normal+vec3(seed.xy,seed.x)));
    float randy = randValNormalDist(rand(normal.zxy*vec3(seed.yx,normal.y)));
    float randz = randValNormalDist(rand(normal.yzx/vec3(seed.x,seed.yx)));
    vec3 randvec = vec3(randx,randy,randz);
    return normalize(randvec*sign(dot(normal,randvec)));
}

vec3 trace(vec3 o, vec3 d, vec2 seed) {
    vec3 O = o;
    vec3 D = d;
    vec3 col = vec3(1);
    vec3 incLight = vec3(0);
    raySceneResult hit = rayScene(O,D);
    if (hit.id < 0) {
        return bgCol(d);
    }
    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        hit = rayScene(O,D);
        if (hit.id > -1) {
            O = hit.loc;
            D = hemisphereSample(hit.norm, seed);
            mtl hitMtl = newMtl(hit.material);
            mtl m = mapMtl(hitMtl,hit.uvSample);
            vec3 emission = m.Ke * m.EmissionStrength;
            incLight += emission * col;
            col *= m.Kd;
        } else {
            break;
        }
    }
    return incLight;
}

vec3 shadePoint(raySceneResult In) {
    vec3 col;
    if (In.id < 0) {
        return bgCol(In.dir);
    }
    mtl M = newMtl(In.material);
    if (M.EmissionStrength > 0) {
        return M.Ke*M.EmissionStrength;
    }
    M = mapMtl(M, In.uvSample);
    col = vec3(0);
    bool didAmbient = false;
    float Nflip = (dot(In.norm, In.dir) > 0 ? -1 : 1);
    vec3 N = normalize(In.norm)*Nflip;
    for (int i = 0; i < numEmissives; i++) {
        float intensity = 0;
        float totalRays = 0;
        float rayHits = 0;
        int emissiveCollectionType = int(EmissiveData[8*i + 1]);
        mtl emissiveMaterial = newMtl(int(EmissiveData[8*i + 2]));
        int emissiveCollectionStart = int(EmissiveData[8*i + 3]);
        int emissiveCollectionEnd = int(EmissiveData[8*i + 4]);
        float emissiveRadius = EmissiveData[8*i + 5];
        vec3 emissiveCenter = vec3(EmissiveData[8*i + 6], EmissiveData[8*i + 7], EmissiveData[8*i + 8]);
        vec3 Nd = normalize(In.loc-emissiveCenter);
        vec3 uvec = normalize(cross(Nd, Nd+vec3(0.1, 31, 1)));
        vec3 vvec = normalize(cross(Nd, uvec));
        if (SAMPLE_RES > 0) {
            if (SAMPLE_RES == 1) {
                vec3 point = emissiveCenter;
                vec3 shootVec = normalize(point-In.loc);
                raySceneResult rayHit = rayScene(In.loc+ 1e-4*N, normalize(shootVec+(rand(vec3(u_seed/424.42, fract(u_seed/645.638), u_seed)))));
                intensity = ((rayHit.type == emissiveCollectionType && rayHit.id >= emissiveCollectionStart && rayHit.id <= emissiveCollectionEnd) ? 1 : 0);
            } else {
                for (int i = 0; i <= SAMPLE_RES; ++i) {
                    float l_u = -1.0 + 2.0 * float(i) / float(SAMPLE_RES);
                    for (int j = 0; j <= SAMPLE_RES; ++j) {
                        float l_v = -1.0 + 2.0 * float(j) / float(SAMPLE_RES);
                        vec3 point = emissiveCenter + emissiveRadius * (l_u * uvec + l_v * vvec);
                        if (length(point - emissiveCenter) <= emissiveRadius*(1 + 1/SAMPLE_RES)) {
                            vec3 shootVec = normalize(point-In.loc);
                            totalRays+=1;
                            raySceneResult rayHit = rayScene(In.loc+ 1e-4*N, normalize(shootVec+(rand(vec3(u_seed/424.42, fract(u_seed/645.638), u_seed)))/2));
                            if (rayHit.type == emissiveCollectionType && rayHit.id >= emissiveCollectionStart && rayHit.id <= emissiveCollectionEnd) {
                                rayHits+=1;
                            }
                        }
                    }
                }
                intensity = emissiveMaterial.EmissionStrength*(rayHits) / float(totalRays);
            }
        } else {
            intensity = 1;
        }
        N = (M.map_norm > -1 ? sampleTexture(M.map_norm, In.uvSample) : N);
        vec3 Id = emissiveMaterial.Ke;
        vec3 diffuse = M.Kd*Id*intensity*clamp(dot(N, normalize(emissiveCenter-In.loc)), 0, 1);
        if (M.illum == 0) {
            col += diffuse;
        }
        if (M.illum == 1) {
            if (!didAmbient) {
                didAmbient = true;
                col += diffuse + M.Ka;
            } else {
                col += diffuse;
            }
        }
        vec3 L = normalize(emissiveCenter - In.loc);
        vec3 V = normalize(-In.dir);
        vec3 R = normalize(2*dot(L, N)*N - L);
        vec3 specular = max(M.Ks * emissiveMaterial.Ks * pow(dot(R, V), M.Ns) * Id * intensity, vec3(0));
        if (M.illum == 2) {
            if (!didAmbient) {
                didAmbient = true;
                col += diffuse + M.Ka + specular;;
            } else {
                col += diffuse + specular;
            }
        }
    }
    return col;
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

    //adjust reflect multiplier for object reflectivity
    ret = (initReflectAmount + (1.0-initReflectAmount) * ret);
    return ret;
}

vec3 pathTrace(vec3 o, vec3 d) {
    ABSORB = vec3(1);
    raySceneResult hit = rayScene(o, d);
    clearIndiceStack();
    addToIndiceStack(1.0029);
    MAT_FILTER = vec3(1);
    RAY_WAS_IN_OBJECT = false;
    if (dot(hit.norm,d) > 0) {
        RAY_IN_OBJECT = true;
        RAY_WAS_IN_OBJECT = true;
        RAY_ENTER_LOCATION = hit.loc;
        mtl M = newMtl(hit.material);
        M = mapMtl(M, hit.uvSample);
        MAT_FILTER = M.Tf;
        addToIndiceStack(M.Ni);
    } else {
        RAY_IN_OBJECT = false;
    }

    vec3 col = vec3(0);
    vec3 throughput = vec3(1);
    for (int bounces = 0; bounces < MAX_BOUNCES; bounces++) {
        //ends if no intersection
        vec3 flatShadeHit = shadePoint(hit);
        if (hit.id < 0) {
            return mix(col, flatShadeHit, throughput);
        }
        hit.dir = normalize(hit.dir);
        //ends if no need to reflect or transmit
        mtl M = newMtl(hit.material);
        M = mapMtl(M, hit.uvSample);
        float Nflip = (dot(hit.norm, hit.dir) > 0 ? -1: 1);
        float fresnelReflFactor = 0;

        float dotNV = dot(hit.norm, hit.dir);

        float eta = 1;
        if (dotNV < 0) {
            RAY_IN_OBJECT = true;
            MAT_FILTER = M.Tf;
            RAY_ENTER_LOCATION = hit.loc;
            addToIndiceStack(M.Ni);
            eta = refractionIndiceStack[1]/refractionIndiceStack[0];
            if (M.Pm > 0 || M.Tr > 0) fresnelReflFactor = fresnelReflectAmount(refractionIndiceStack[1], refractionIndiceStack[0], hit.norm*Nflip, hit.dir, M.Pm);
        } else {
            RAY_IN_OBJECT = false;
            eta = refractionIndiceStack[0]/refractionIndiceStack[1];
            if (M.Pm > 0 || M.Tr > 0) fresnelReflFactor = fresnelReflectAmount(refractionIndiceStack[0], refractionIndiceStack[1], hit.norm*Nflip, hit.dir, M.Pm);
            removeFirstOfIndiceStack();
        }

        if (RAY_WAS_IN_OBJECT && !RAY_IN_OBJECT) {
            float dist = distance(RAY_ENTER_LOCATION,hit.loc);
            ABSORB = exp((-1/MAT_FILTER)*dist);
        } else {
            ABSORB = vec3(1);
        }
        throughput *= ABSORB;

        float reflCo = fresnelReflFactor;
        float transCo = M.Tr * (1-fresnelReflFactor);
        if (reflCo + transCo == 0) {
            return mix(col, flatShadeHit, throughput);
        }
        if (reflCo>transCo) /*do 1 refract ray, follow reflect ray*/{
            if (transCo > 0) {
                bool TEMP_RAY_IN_OBJ = false;
                float eta = 1;
                if (dotNV < 0) {
                    addToIndiceStack(M.Ni);
                    eta = refractionIndiceStack[1]/refractionIndiceStack[0];
                    TEMP_RAY_IN_OBJ = true;
                } else {
                    eta = refractionIndiceStack[0]/refractionIndiceStack[1];
                    removeFirstOfIndiceStack();
                }
                vec3 refractedRay = refract(hit.dir, hit.norm*Nflip, eta);
                raySceneResult currRefrRayHit = rayScene(hit.loc - hit.norm*Nflip*1e-4, refractedRay);
                col = mix(col, mix(flatShadeHit, shadePoint(currRefrRayHit), transCo), throughput);
                //if the refraction ray was entering, and its hit surface would let it exit...
                mtl M_refr = newMtl(currRefrRayHit.material);
                M_refr = mapMtl(M_refr, currRefrRayHit.uvSample);
                /*
                if (TEMP_RAY_IN_OBJ) {
                    return vec3(1,0,0);
                    float currFrenselReflFactor = 0;
                    if (M.Tr > 0) currFrenselReflFactor = fresnelReflectAmount(refractionIndiceStack[0], M.Ni, hit.norm*Nflip, hit.dir, M.Pm);
                    float currTransCo = M.Tr * (1-currFrenselReflFactor);
                    int subBounceCount = 0;
                    vec3 tempThroughput = throughput;
                    while (currTransCo > 0.1 && subBounceCount < MAX_BOUNCES - bounces) {
                        if (length(tempThroughput) < 0.01) break;
                        subBounceCount++;

                    }
                }
                */
            } else {
                col = mix(col, flatShadeHit, throughput);
            }
            hit = rayScene(hit.loc + hit.norm*Nflip*1e-4, reflect(hit.dir, hit.norm*Nflip));
            throughput = throughput*(reflCo);
        } else /*do 1 reflect ray, follow refract ray*/{
            if (reflCo > 0) {
                raySceneResult refldRay = rayScene(hit.loc + hit.norm*Nflip*1e-6, reflect(hit.dir, hit.norm*Nflip));
                col = mix(col, mix(flatShadeHit, shadePoint(refldRay), reflCo), throughput);
            } else {
                col = mix(col, flatShadeHit, throughput);
            }
            vec3 refractedRay = refract(hit.dir, hit.norm*Nflip, eta);
            vec3 randomPerturb = vec3(0);
            if (M.Pr > 0) randomPerturb = M.Pr * vec3(rand(hit.loc)*2 - 1, rand(hit.loc+hit.dir)*2 - 1, rand(hit.loc+2*hit.dir+hit.id)*2 -1);
            hit = rayScene(hit.loc - hit.norm*Nflip*1e-4, normalize(refractedRay + randomPerturb));
            throughput = throughput*vec3(transCo);
        }
        RAY_WAS_IN_OBJECT = RAY_IN_OBJECT;
    }
    return col;
}

void main() {
    vec3 norm = vec3(0, 0, 0);
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    if (pixel_coords.x >= int(resolution) || pixel_coords.y >= int(resolution * screenHratio)) return;
    uint index = pixel_coords.y * uint(resolution) + pixel_coords.x;
    vec3 direction = directions[index];
    vec3 col = pathTrace(origin, direction+(rand(vec3(u_seed/424.42,fract(u_seed/645.638),u_seed)-vec3(0.5))/1000))-vec3(GAMMA);
    vec4 finalColor = vec4(col, 1.0);
    vec4 lastFrameColor = imageLoad(FRAME, pixel_coords);
    vec4 averagedColor = ((lastFrameColor * float(u_frameCount)) + vec4(col, 1)) / float(u_frameCount+1);
    imageStore(FRAME, pixel_coords, averagedColor);
}