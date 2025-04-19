package Main;

import org.lwjgl.BufferUtils;
import org.lwjgl.opengl.GL;
import org.lwjgl.system.MemoryUtil;

import javax.imageio.ImageIO;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.*;
import java.lang.reflect.Field;
import java.nio.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static Main.dispatch.material.parseMtls;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.ARBBindlessTexture.*;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL42.*;
import static org.lwjgl.opengl.GL43.GL_COMPUTE_SHADER;
import static org.lwjgl.opengl.GL43.GL_SHADER_STORAGE_BUFFER;
import static org.lwjgl.opengl.GL45.*;
import static org.lwjgl.stb.STBImage.stbi_image_free;
import static org.lwjgl.stb.STBImage.stbi_load;
import static org.lwjgl.system.MemoryUtil.NULL;


public class dispatch {
    //SETUP VARS / SETTINGS + LIST INITIATION
    //<editor-fold desc = "setup vars (screen, cam) collapsed">
    private static final boolean REALTIME = true;
    private static final int autoSC = -1;

    public static final int MAX_BVH_BRANCHES = 256;
    public static final int MAX_TRIS_IN_BVH_LEAF = 1;
    public static final int OPTIMIZATION_LEVEL = 5;

    private static final int WIDTH = 1920;
    private static final int HEIGHT = 1080;
    private static final int res = 1920;
    private static final boolean RAYTRACING = true;
    private static final boolean DEBUG = false;
    private static final int SAMPLE_RESOLUTION = 8;
    private static final int MAX_BOUNCES = 6;
    private static final float NEGATIVE_GAMMA = 0.0f;
    private static final float BLUR_STRENGTH = 0.001f;
    private static final float FOCAL_DISTANCE = 1f;
    private static final boolean AUTO_FOCUS = true;
    private static final float camSize = 1.5f;
    private static final float focalLength = 1;
    //box demo
    private static final float[] cam = {(float) 0, (float) .5, (float) 0};
    private static final float[] rot = {(float) 0.25, (float) 0, (float) 0};
    //inputs
    private static float MOVE_SPEED = 0.1f;
    private static final float sensitivity = 1.25f;

    private static final float screenHratio = HEIGHT / (float) WIDTH;

    public static class VARS {
        private static long lastFPSCheck = 0;
        private static int frameCount = 0;
        private static int currentFPS = 0;
        private static long lastFrame = System.nanoTime();
        private static long lastInFrame = 0;
        private static final boolean SHOW_FPS_IN_TITLE = true;
        private static boolean CAM_MOVING = false;
        private static boolean RECALC_VECTORS = true;
        private static float[] lastCam = cam;
        private static float[] lastRot = rot;
    }
    //</editor-fold>

    //<editor-fold desc = "setup vars (scene) collapsed">
    //Specular is specular reflection color, diffuse is lambertian reflectance, ambient is general
    //Ks, Kd, Ka, alpha, reflectivity, opacity, refractive index
    private static final List<List<Float>> mats = new ArrayList<>();
    private static List<String> textures = new ArrayList<>();
    private static final List<String> textureNames = new ArrayList<>();
    private static final int NUM_MATERIAL_PARAMETERS = 48;
    private static final List<material> materials = new ArrayList<>();

    //OBJECTS
    //tris (all data)
    //verts
    private static int NEXT_TRI_ID = 0;
    private static final List<triangle> triangles = new ArrayList<>();

    //implicits
    private static final List<Integer> fn = new ArrayList<>();
    private static final List<vec> Ishift = new ArrayList<>();
    private static final List<vec> Irot = new ArrayList<>();
    private static final List<vec> Iscale = new ArrayList<>();
    private static final List<Integer> Im = new ArrayList<>();

    //ellipsoids
    private static final List<vec> Ec = new ArrayList<>();
    private static final List<vec> Estretch = new ArrayList<>();
    private static final List<vec> Erot = new ArrayList<>();
    private static final List<Float> Erad = new ArrayList<>();
    private static final List<Integer> Em = new ArrayList<>();

    private static final List<Long> textureHandles = new ArrayList<>();
    //</editor-fold>

    //<editor-fold desc = "setup BVH lists collapsed">
    // corner 1 = 0,1,2, corner 2 = 3,4,5, type = 6, ID collection start = 7, ID collection end = 8
    private static final List<BVH> sceneObjs = new ArrayList<>();
    private static final List<Float> sceneBVHdata = new ArrayList<>();
    private static final List<Integer> sceneBVHtree = new ArrayList<>();
    private static final List<Integer> BVHleafTriIndices = new ArrayList<>();
    private static final List<Integer> objIndicesInTree = new ArrayList<>();

    private static int nextBVHId = 0;
    //</editor-fold>

    //<editor-fold desc = "screen gen shader (vert+frag) collapsed">
    private static final String QUAD_VERTEX_SHADER;
    static {
        try {
            QUAD_VERTEX_SHADER = functions.getString("C:\\Users\\enzo\\IdeaProjects\\raytracer\\src\\shaders\\vert.glsl");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static final String QUAD_FRAGMENT_SHADER;
    static {
        try {
            QUAD_FRAGMENT_SHADER = functions.getString("C:\\Users\\enzo\\IdeaProjects\\raytracer\\src\\shaders\\frag.glsl");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    //</editor-fold>

    //EXECUTE + DEFINE SCENE
    public static void main(String[] args) throws Exception {
        if (!glfwInit()) {
            throw new IllegalStateException("Unable to initialize GLFW");
        }


        //<editor-fold desc = "render init collapsed">


        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        long window = glfwCreateWindow(WIDTH, HEIGHT, "Ray Tracer", NULL, NULL);
        glfwMakeContextCurrent(window);
        GL.createCapabilities();
        int quadVertexShader = createShader(GL_VERTEX_SHADER, QUAD_VERTEX_SHADER);
        System.out.println("compiling GLSL...");
        long start = System.nanoTime();
        int quadFragmentShader = createShader(GL_FRAGMENT_SHADER, QUAD_FRAGMENT_SHADER);
        glFinish();
        System.out.println("        (took " + ((System.nanoTime() - start) / 1000000000.0) + " seconds)");

        int raytrace = glCreateProgram();
        glAttachShader(raytrace, quadVertexShader);
        glAttachShader(raytrace, quadFragmentShader);
        glLinkProgram(raytrace);
        checkProgramError(raytrace);
        int quadVAO = glGenVertexArrays();
        glBindVertexArray(quadVAO);

        glUseProgram(raytrace);
        int FRAME = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, FRAME);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, res, (int) (res * screenHratio));
        glBindImageTexture(0, FRAME, 0, false, 0, GL_READ_WRITE, GL_RGBA32F);

        FloatBuffer paramsBuffer = BufferUtils.createFloatBuffer(14);
        paramsBuffer.put(new float[]{
                camSize,
                focalLength,
                res,
                screenHratio,
                SAMPLE_RESOLUTION,
                MAX_BOUNCES,
                NEGATIVE_GAMMA,
                BLUR_STRENGTH,
                FOCAL_DISTANCE,
                RAYTRACING ? 1f : 0f,
                DEBUG ? 1f : 0f,
                AUTO_FOCUS ? 1f : 0f,
        });
        paramsBuffer.flip();

        int paramsSSBO = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, paramsSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, paramsBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, paramsSSBO);
        //</editor-fold>

        //<editor-fold desc = "scene definition+init collapsed">

        //<editor-fold desc = "scene generation">
        //Ka, Kd, Ks, alpha, reflectivity (0-1, %), transmission (0-1, %), refractive index, emission (0-n), radius of emission, solid? or surface, absorption
        //for emissive materials of meshes, dont assign any other list-adjacent tris to the same material
        //bottles
        System.out.println("creating scene and parsing objects...");
        textures.add("C:\\Graphics\\antiDoxxFolder\\thatch_chapel_4k.png");
        textureNames.add("skybox.png");

        scene.addMaterial("default");
        scene.setLastMtl("Kd", new vec(0.8));
        scene.setLastMtl("Pr", 1);

        scene.addMaterial("test");
        scene.setLastMtl("Kd", new vec(0.8, 0.45, 0.5));
        scene.setLastMtl("Ks", new vec(0.5));
        scene.setLastMtl("Ni", 1.45);
        scene.setLastMtl("Pr", 1);
        scene.setLastMtl("Pc", 0.0);
        scene.setLastMtl("Pcr", 0.0);
        scene.setLastMtl("Tr", 0.7);
        scene.setLastMtl("subsurface", 0);
        scene.setLastMtl("subsurfaceColor", new vec(0.45, 0.8, 0.5));
        scene.setLastMtl("subsurfaceRadius", new vec(1,1,1));
        scene.setLastMtl("Density", 0.1);

        System.out.println("    adding objects and generating bounding volume hierarchies...");
        start = System.nanoTime();

        //scene.addTri(new vec(0), new vec(1,1,0), new vec(0,0,1), 0);
        //scene.addEllipsoid(new vec(0, 0.1, 0.2), new vec(1), new vec(0), 0.05f, 1);
        scene.addObject("C:\\Graphics\\antiDoxxFolder\\box2", 1, new vec(1), new vec(0), new vec(0));
        //scene.addObject("C:\\Graphics\\antiDoxxFolder\\tomato", 1, new vec (0.06), new vec(0,0.0,1.3), new vec(0, 0, 0));
        scene.addObject("C:\\Graphics\\antiDoxxFolder\\dragonSmall", 1, new vec (-1,1,-1), new vec(0,0.3,1), new vec(0, 0, 0));
        //scene.addObject("C:\\Graphics\\antiDoxxFolder\\bust", 1 , new vec(-1.5,1.5,-1.5), new vec(0,0,1), new vec(0));


        //scene.addObject("C:\\Graphics\\antiDoxxFolder\\INTERIOR2", 0, new vec(10), new vec(7, 2.02, 0), new vec(0, 0, 0));
        //scene.addObject("C:\\Graphics\\antiDoxxFolder\\wineglassfull", 1, new vec(0.3), new vec(-4, 2.9, 25), new vec(0, 0, 0));



        System.out.println("    (took " + ((System.nanoTime() - start) / 1000000000.0) + " seconds)");


        //scene.addEllipsoid(new vec(0,1,0),new vec(1), new vec(0), 0.3f, 1);
        //</editor-fold>


        // 0 = null, 1 = tri, 2 = implicit, 3 = ellipsoid
        //<editor-fold desc = "texture and mtl buffers">
        FloatBuffer MTLbuffer = BufferUtils.createFloatBuffer(1 + NUM_MATERIAL_PARAMETERS * materials.size());
        MTLbuffer.put(NUM_MATERIAL_PARAMETERS);
        for (material mtl : materials) {
            MTLbuffer.put((float) mtl.Ka.x);
            MTLbuffer.put((float) mtl.Ka.y);
            MTLbuffer.put((float) mtl.Ka.z);
            MTLbuffer.put((float) mtl.Kd.x);
            MTLbuffer.put((float) mtl.Kd.y);
            MTLbuffer.put((float) mtl.Kd.z);
            MTLbuffer.put((float) mtl.Ks.x);
            MTLbuffer.put((float) mtl.Ks.y);
            MTLbuffer.put((float) mtl.Ks.z);
            MTLbuffer.put((float) mtl.Ns);
            MTLbuffer.put((float) mtl.d);
            MTLbuffer.put((float) mtl.Tr);
            MTLbuffer.put((float) mtl.Tf.x);
            MTLbuffer.put((float) mtl.Tf.y);
            MTLbuffer.put((float) mtl.Tf.z);
            MTLbuffer.put((float) mtl.Ni);
            MTLbuffer.put((float) mtl.Ke.x);
            MTLbuffer.put((float) mtl.Ke.y);
            MTLbuffer.put((float) mtl.Ke.z);
            //density is custom
            MTLbuffer.put((float) mtl.Density);
            MTLbuffer.put((float) mtl.illum);
            MTLbuffer.put((float) mtl.map_Ka);
            MTLbuffer.put((float) mtl.map_Kd);
            MTLbuffer.put((float) mtl.map_Ks);
            //PBR types
            MTLbuffer.put((float) mtl.Pm);
            MTLbuffer.put((float) mtl.Pr);
            MTLbuffer.put((float) mtl.Ps);
            MTLbuffer.put((float) mtl.Pc);
            MTLbuffer.put((float) mtl.Pcr);
            MTLbuffer.put((float) mtl.aniso);
            MTLbuffer.put((float) mtl.anisor);
            MTLbuffer.put((float) mtl.map_Pm);
            MTLbuffer.put((float) mtl.map_Pr);
            MTLbuffer.put((float) mtl.map_Ps);
            MTLbuffer.put((float) mtl.map_Pc);
            MTLbuffer.put((float) mtl.map_Pcr);
            MTLbuffer.put((float) mtl.map_bump);
            MTLbuffer.put((float) mtl.map_d);
            MTLbuffer.put((float) mtl.map_Tr);
            MTLbuffer.put((float) mtl.map_Ns);
            MTLbuffer.put((float) mtl.map_Ke);
            // Custom
            MTLbuffer.put((float) mtl.subsurface);
            MTLbuffer.put((float) mtl.subsurfaceColor.x);
            MTLbuffer.put((float) mtl.subsurfaceColor.y);
            MTLbuffer.put((float) mtl.subsurfaceColor.z);
            MTLbuffer.put((float) mtl.subsurfaceRadius.x);
            MTLbuffer.put((float) mtl.subsurfaceRadius.y);
            MTLbuffer.put((float) mtl.subsurfaceRadius.z);
        }
        MTLbuffer.flip();
        int mtlBuffer = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mtlBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, MTLbuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, mtlBuffer);

        System.out.println("uploading textures and materials...");
        int counter = 0;
        start = System.nanoTime();
        for (String path : textures) {
            counter++;
            System.out.print("\r    uploading (" + counter + "/" + textures.size() + "), " + path + ", ...");
            int textureID = glCreateTextures(GL_TEXTURE_2D);

            // Load image data
            IntBuffer width = MemoryUtil.memAllocInt(1);
            IntBuffer height = MemoryUtil.memAllocInt(1);
            IntBuffer channels = MemoryUtil.memAllocInt(1);
            ByteBuffer image = stbi_load(path, width, height, channels, 4);
            if (image == null) {
                System.err.println("Failed to load texture: " + path);
                continue;
            }

            glTextureStorage2D(textureID, 1, GL_RGBA8, width.get(0), height.get(0));
            glTextureSubImage2D(textureID, 0, 0, 0, width.get(0), height.get(0), GL_RGBA, GL_UNSIGNED_BYTE, image);
            glTextureParameteri(textureID, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(textureID, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

            stbi_image_free(image);
            MemoryUtil.memFree(width);
            MemoryUtil.memFree(height);
            MemoryUtil.memFree(channels);

            // Make the texture bindless
            long handle = glGetTextureHandleARB(textureID);
            glMakeTextureHandleResidentARB(handle);
            textureHandles.add(handle);
            if (!glIsTextureHandleResidentARB(handle)) {
                System.err.println("Texture handle is not resident: " + handle);
            }
        }
        System.out.println();
        LongBuffer handlesBuffer = MemoryUtil.memAllocLong(textures.size());
        for (Long handle : textureHandles) {
            handlesBuffer.put(handle);
        }
        int handleBuffer = glGenBuffers();
        handlesBuffer.flip();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, handleBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, handlesBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, handleBuffer);
        System.out.println("    (took " + ((System.nanoTime() - start) / 1000000000.0) + " seconds)");

        //</editor-fold>

        //<editor-fold desc = "tris buffer">
        System.out.println("uploading tris...");
        start = System.nanoTime();
        FloatBuffer TriBuffer = BufferUtils.createFloatBuffer(40*triangles.size());
        counter = 0;
        for (triangle tri : triangles) {
            TriBuffer.put((float)tri.v1.x).put((float)tri.v1.y).put((float)tri.v1.z).put(0.0f);
            TriBuffer.put((float)tri.v2.x).put((float)tri.v2.y).put((float)tri.v2.z).put(0.0f);
            TriBuffer.put((float)tri.v3.x).put((float)tri.v3.y).put((float)tri.v3.z).put(0.0f);

            if (!(tri.n1.x == 0 && tri.n1.y == 0 && tri.n1.z == 0)) {
                TriBuffer.put((float)tri.n1.x).put((float)tri.n1.y).put((float)tri.n1.z).put(0.0f);
                TriBuffer.put((float)tri.n2.x).put((float)tri.n2.y).put((float)tri.n2.z).put(0.0f);
                TriBuffer.put((float)tri.n3.x).put((float)tri.n3.y).put((float)tri.n3.z).put(0.0f);
            } else {
                vec norm = (tri.v3.sub(tri.v1)).cross(tri.v2.sub(tri.v1));
                TriBuffer.put((float)norm.x).put((float)norm.y).put((float)norm.z).put(0.0f);
                TriBuffer.put(0.0f).put(0.0f).put(0.0f).put(0.0f);
                TriBuffer.put(0.0f).put(0.0f).put(0.0f).put(0.0f);
            }

            if (tri.vt1.x != 69.420 && tri.vt1.y != 0) {
                TriBuffer.put((float)tri.vt1.x).put((float)tri.vt1.y).put(0.0f).put(0.0f);
                TriBuffer.put((float)tri.vt2.x).put((float)tri.vt2.y).put(0.0f).put(0.0f);
                TriBuffer.put((float)tri.vt3.x).put((float)tri.vt3.y).put(0.0f).put(0.0f);
            } else {
                TriBuffer.put(69.420f).put(0.0f).put(0.0f).put(0.0f);
                TriBuffer.put(0.0f).put(0.0f).put(0.0f).put(0.0f);
                TriBuffer.put(0.0f).put(0.0f).put(0.0f).put(0.0f);
            }

            TriBuffer.put((float)tri.material).put(0.0f).put(0.0f).put(0.0f);

            counter++;
            System.out.print("\r    uploaded (" + counter + "/" + triangles.size() + ") triangles");
        }
        System.out.println();
        TriBuffer.flip();
        int triangleBuffer = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, triangleBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, TriBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, triangleBuffer);
        System.out.println("    (took " + ((System.nanoTime() - start) / 1000000000.0) + " seconds)");
        //</editor-fold>

        //<editor-fold desc = "implicits buffer">
        FloatBuffer ImplicitBuffer = BufferUtils.createFloatBuffer(1 + fn.size() + Ishift.size() * 3 + Iscale.size() * 3 + Irot.size() * 3 + Im.size());
        ImplicitBuffer.put((float) fn.size());
        for (int Fn : fn) {
            ImplicitBuffer.put((float) Fn);
        }
        for (vec shift : Ishift) {
            ImplicitBuffer.put((float) shift.x);
            ImplicitBuffer.put((float) shift.y);
            ImplicitBuffer.put((float) shift.z);
        }
        for (vec scale : Iscale) {
            ImplicitBuffer.put((float) scale.x);
            ImplicitBuffer.put((float) scale.y);
            ImplicitBuffer.put((float) scale.z);
        }
        for (vec rot : Irot) {
            ImplicitBuffer.put((float) rot.x);
            ImplicitBuffer.put((float) rot.y);
            ImplicitBuffer.put((float) rot.z);
        }
        for (int m : Im) {
            ImplicitBuffer.put((float) m);
        }
        ImplicitBuffer.flip();
        int implicitBuffer = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, implicitBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, ImplicitBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, implicitBuffer);
        //</editor-fold desc>

        //<editor-fold desc = "ellipsoids buffer">
        FloatBuffer EllipsoidBuffer = BufferUtils.createFloatBuffer(1 + Ec.size() * 3 + Estretch.size() * 3 + Erot.size() * 3 + Erad.size() + Em.size());
        EllipsoidBuffer.put((float) Ec.size());
        for (vec center : Ec) {
            EllipsoidBuffer.put((float) center.x);
            EllipsoidBuffer.put((float) center.y);
            EllipsoidBuffer.put((float) center.z);
        }
        for (vec stretch : Estretch) {
            EllipsoidBuffer.put((float) stretch.x);
            EllipsoidBuffer.put((float) stretch.y);
            EllipsoidBuffer.put((float) stretch.z);
        }
        for (vec rot : Erot) {
            EllipsoidBuffer.put((float) rot.x);
            EllipsoidBuffer.put((float) rot.y);
            EllipsoidBuffer.put((float) rot.z);
        }
        for (float rad : Erad) {
            EllipsoidBuffer.put((float) rad);
        }
        for (int m : Em) {
            EllipsoidBuffer.put((float) m);
        }
        EllipsoidBuffer.flip();
        int ellipsoidData = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ellipsoidData);
        glBufferData(GL_SHADER_STORAGE_BUFFER, EllipsoidBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ellipsoidData);
        //</editor-fold>

        System.out.println("flattening bounding volume hierarchies...");
        start = System.nanoTime();
        BVH.allBVHtoList();
        System.out.println("    (took " + ((System.nanoTime() - start) / 1000000000.0) + " seconds)");
        BVH.sortTree();
        //<editor-fold desc = "BVH buffer">
        FloatBuffer BVHdatabuffer = BufferUtils.createFloatBuffer(sceneBVHdata.size());
        for (Float sceneBVHdata : sceneBVHdata) {
            BVHdatabuffer.put(sceneBVHdata);
        }
        BVHdatabuffer.flip();
        int BvhData = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, BvhData);
        glBufferData(GL_SHADER_STORAGE_BUFFER, BVHdatabuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, BvhData);
        IntBuffer BVHtreebuffer = BufferUtils.createIntBuffer(sceneBVHtree.size());
        for (int sceneBVHtree : sceneBVHtree) {
            BVHtreebuffer.put(sceneBVHtree);
        }
        BVHtreebuffer.flip();
        int BVHtree = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, BVHtree);
        glBufferData(GL_SHADER_STORAGE_BUFFER, BVHtreebuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, BVHtree);

        IntBuffer leafTriIndicesBuffer = BufferUtils.createIntBuffer(BVHleafTriIndices.size());
        for (int index : BVHleafTriIndices) {
            leafTriIndicesBuffer.put(index);
        }
        leafTriIndicesBuffer.flip();
        int BVHleafTriIndices = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, BVHleafTriIndices);
        glBufferData(GL_SHADER_STORAGE_BUFFER, leafTriIndicesBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, BVHleafTriIndices);

        IntBuffer objIndicesBuffer = BufferUtils.createIntBuffer(objIndicesInTree.size() + 1);
        objIndicesBuffer.put(objIndicesInTree.size());
        for (int objIndex : objIndicesInTree) {
            objIndicesBuffer.put(objIndex);
        }
        objIndicesBuffer.flip();
        int objIndicesbuffer = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, objIndicesbuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, objIndicesBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, objIndicesbuffer);
        //</editor-fold>

        //includes emissive collection determination
        //<editor-fold desc = "materials buffer">
        FloatBuffer materialBuffer = BufferUtils.createFloatBuffer(1 + 18 * mats.size());
        materialBuffer.put(mats.size());
        for (List<Float> mat : mats) {
            for (int j = 0; j < 18; j++) {
                materialBuffer.put(mat.get(j));
            }
        }
        materialBuffer.flip();
        int matBuffer = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, matBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, materialBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, matBuffer);
        //</editor-fold>

        //<editor-fold desc = "other initiation">
        FloatBuffer camBuffer = BufferUtils.createFloatBuffer(3);
        camBuffer.put(cam);
        camBuffer.flip();
        int camSSBO = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, camSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, camBuffer, GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, camSSBO);

        FloatBuffer rotBuffer = BufferUtils.createFloatBuffer(3);
        rotBuffer.put(rot);
        rotBuffer.flip();
        int rotationSSBO = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, rotationSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 3 * Float.BYTES, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, rotationSSBO);

        FloatBuffer mouseBuffer = BufferUtils.createFloatBuffer(3);
        int mouseSSBO = glGenBuffers();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, mouseSSBO);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 3*Float.BYTES, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mouseSSBO);

        glBindTexture(GL_TEXTURE_2D, FRAME);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_ACCUM);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_ACCUM);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glUseProgram(raytrace);
        int texLocation = glGetUniformLocation(raytrace, "FRAME");
        glUniform1i(texLocation, 0);


        System.out.println("Commands: \"telemetry\"" + ", \"screenshot\"");
        VARS.lastFPSCheck = System.currentTimeMillis();
        //</editor-fold>

        boolean was_moving;
        boolean did_regress = true;
        int FRAMES_STILL = 0;
        while (!glfwWindowShouldClose(window)) {
            if (autoSC != -1 && FRAMES_STILL >= autoSC && RAYTRACING && !DEBUG && VARS.currentFPS < 10) {
                LocalDateTime now = LocalDateTime.now();
                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy_MM_dd_HH_mm");
                String formattedDateTime = now.format(formatter);
                glFinish();
                functions.screenshot("autoScreenshot_" + (FRAMES_STILL-autoSC) + "_" + formattedDateTime + ".png");
                glFinish();
                Thread.sleep(1000);
            }
            if (REALTIME) {
                VARS.frameCount++;
                functions.commands(FRAME);
                long currentTime = System.currentTimeMillis();
                if (currentTime - VARS.lastFPSCheck >= 1000) {
                    VARS.currentFPS = VARS.frameCount;
                    VARS.frameCount = 0;
                    VARS.lastFPSCheck = currentTime;
                    if (VARS.currentFPS > 4000) {
                        System.out.println("Crashed!");
                        glfwSetWindowShouldClose(window, true);
                        break;
                    }
                    if (VARS.SHOW_FPS_IN_TITLE) {
                        glfwSetWindowTitle(window, "Ray Tracer - FPS: " + VARS.currentFPS + " (" + FRAMES_STILL + " frames rendered)" + ", Position - X: " + Math.floor(cam[0] * 1000) / 1000 + ", Y: " + Math.floor(cam[1] * 1000) / 1000 + ", Z: " + Math.floor(cam[2] * 1000) / 1000);
                    }
                }

                was_moving = VARS.CAM_MOVING;
                functions.move(window);
                VARS.RECALC_VECTORS = !(Arrays.equals(VARS.lastRot, rot));
                VARS.CAM_MOVING = !(Arrays.equals(VARS.lastCam, cam) && Arrays.equals(VARS.lastRot, rot));
                VARS.lastCam = cam.clone();
                VARS.lastRot = rot.clone();

                camBuffer.clear();
                camBuffer.put(cam).flip();
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, camSSBO);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, camBuffer);
                rotBuffer.clear();
                rotBuffer.put(rot).flip();
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, rotationSSBO);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, rotBuffer);
                vec mousePos = functions.getCursorPos(window);
                mouseBuffer.clear();
                mouseBuffer.put((float) mousePos.x);
                mouseBuffer.put((float) (res*screenHratio - mousePos.y));
                mouseBuffer.put(glfwGetMouseButton(window,0));
                mouseBuffer.flip();
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, mouseSSBO);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, mouseBuffer);


                if (VARS.CAM_MOVING || was_moving || FRAMES_STILL == -1) {
                    FRAMES_STILL = 0;
                    paramsBuffer.clear();
                    paramsBuffer.put(new float[]{
                            camSize,
                            focalLength,
                            500,
                            screenHratio,
                            4,
                            2,
                            NEGATIVE_GAMMA,
                            BLUR_STRENGTH,
                            FOCAL_DISTANCE,
                            RAYTRACING ? 1f : 0f,
                            DEBUG ? 1f : 0f,
                            AUTO_FOCUS ? 1f : 0f
                    });
                    paramsBuffer.flip();
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, paramsSSBO);
                    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, paramsBuffer);
                    resetTexture(FRAME);
                    did_regress = false;
                } else {
                    if (!did_regress) {
                        did_regress = true;
                        paramsBuffer.clear();
                        paramsBuffer.put(new float[]{
                                camSize,
                                focalLength,
                                res,
                                screenHratio,
                                SAMPLE_RESOLUTION,
                                MAX_BOUNCES,
                                NEGATIVE_GAMMA,
                                BLUR_STRENGTH,
                                FOCAL_DISTANCE,
                                RAYTRACING ? 1f : 0f,
                                DEBUG ? 1f : 0f,
                                AUTO_FOCUS ? 1f : 0f
                        });
                        paramsBuffer.flip();
                        glBindBuffer(GL_SHADER_STORAGE_BUFFER, paramsSSBO);
                        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, paramsBuffer);
                        resetTexture(FRAME);
                    }
                }

                glUseProgram(raytrace);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, FRAME);
                glBindImageTexture(0, FRAME, 0, false, 0, GL_READ_WRITE, GL_RGBA32F);
                glUniform1i(glGetUniformLocation(raytrace, "u_frameCount"), FRAMES_STILL);
                glUniform1i(glGetUniformLocation(raytrace, "u_seed"), (int)(Math.random()*10000));
                glMemoryBarrier(GL_ALL_BARRIER_BITS);

                glClear(GL_COLOR_BUFFER_BIT);
                glBindVertexArray(quadVAO);

            }
            glDrawArrays(GL_TRIANGLES, 0, 6);
            functions.commands(FRAME);

            FRAMES_STILL++;

            glfwSwapBuffers(window);
            glfwPollEvents();

        }
        //<editor-fold desc = "finish">
        glDeleteTextures(FRAME);
        glDeleteBuffers(new int[]{
                paramsSSBO, rotationSSBO, camSSBO
        });
        glDeleteShader(raytrace);
        glDeleteShader(paramsSSBO);
        glDeleteVertexArrays(quadVAO);
        glDeleteShader(quadVertexShader);
        glDeleteShader(quadFragmentShader);
        glfwDestroyWindow(window);
        glfwTerminate();
        //</editor-fold>
    }

    //UTIL FUNCTIONS AND DATATYPE CREATION
    //render util

    static void resetTexture(int LAST_FRAME) {
        glBindImageTexture(0, LAST_FRAME, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);
        glClearTexImage(LAST_FRAME, 0, GL_RGBA, GL_FLOAT, (ByteBuffer) null); // Clear texture
    }

    public static class functions {
        public static void move(long window) {
            long currentTime = System.nanoTime();
            double deltaTime = (currentTime - VARS.lastFrame) / 1_000_000_000.0;
            VARS.lastFrame = currentTime;
            deltaTime = Math.min(deltaTime, 0.1);

            float moveSpeed = MOVE_SPEED * (float) deltaTime;
            float sense = sensitivity * (float) deltaTime;

            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) rot[0] -= sense;
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) rot[0] += sense;
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) rot[1] += sense;
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) rot[1] -= sense;
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                cam[0] -= (float) (moveSpeed * Math.cos(rot[1] + Math.PI / 2));
                cam[2] += (float) (moveSpeed * Math.sin(rot[1] + Math.PI / 2));
            }
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                cam[0] += (float) (moveSpeed * Math.cos(rot[1]));
                cam[2] -= (float) (moveSpeed * Math.sin(rot[1]));
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                cam[0] += (float) (moveSpeed * Math.cos(rot[1] + Math.PI / 2));
                cam[2] -= (float) (moveSpeed * Math.sin(rot[1] + Math.PI / 2));
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                cam[0] -= (float) (moveSpeed * Math.cos(rot[1]));
                cam[2] += (float) (moveSpeed * Math.sin(rot[1]));
            }
            if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS && System.currentTimeMillis() > VARS.lastInFrame + 500) {
                MOVE_SPEED *= 10;
                VARS.lastInFrame = System.currentTimeMillis();
            }
            if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS && System.currentTimeMillis() > VARS.lastInFrame + 500) {
                MOVE_SPEED /= 10;
                VARS.lastInFrame = System.currentTimeMillis();
            }
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) cam[1] -= moveSpeed;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) cam[1] += moveSpeed;
        }

        public static void commands(int screen) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            try {
                if (reader.ready()) {
                    String input = reader.readLine();
                    if (input != null) {
                        if (input.contains("telemetry")) {
                            vec CameraPos = new vec(cam[0], cam[1], cam[2]);
                            vec Rotation = new vec(rot[0], rot[1], rot[2]);
                            System.out.println("Camera Position: ");
                            CameraPos.println();
                            System.out.println("Rotation:");
                            Rotation.println();
                        }
                        if (input.contains("screenshot")) {
                            glFinish();
                            screenshot("GLSL_render.png");
                        }
                    }
                }
            } catch (Exception e) {
                System.err.println("Console input error: " + e.getMessage());
            }
        }

        public static void screenshot(String fileName) {
            // Create a buffer to hold the texture data
            //Creating an rbg array of total pixels
            int[] pixels = new int[WIDTH * HEIGHT];
            int bindex;
            // allocate space for RBG pixels
            ByteBuffer fb = ByteBuffer.allocateDirect(WIDTH * HEIGHT * 3);

            // grab a copy of the current frame contents as RGB
            glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, fb);

            BufferedImage imageIn = new BufferedImage(WIDTH, HEIGHT,BufferedImage.TYPE_INT_RGB);
            // convert RGB data in ByteBuffer to integer array
            for (int i=0; i < pixels.length; i++) {
                bindex = i * 3;
                pixels[i] =
                        ((fb.get(bindex) << 16))  +
                                ((fb.get(bindex+1) << 8))  +
                                ((fb.get(bindex+2) << 0));
            }
            //Allocate colored pixel to buffered Image
            imageIn.setRGB(0, 0, WIDTH, HEIGHT, pixels, 0 , WIDTH);

            //Creating the transformation direction (horizontal)
            AffineTransform at =  AffineTransform.getScaleInstance(1, -1);
            at.translate(0, -imageIn.getHeight(null));

            //Applying transformation
            AffineTransformOp opRotated = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
            BufferedImage imageOut = opRotated.filter(imageIn, null);

            try {
                // Create screenshots directory if it doesn't exist
                File outputDir = new File("screenshots");
                if (!outputDir.exists()) {
                    outputDir.mkdir();
                }

                // Save the image
                File outputFile = new File(outputDir, fileName);
                ImageIO.write(imageOut, "PNG", outputFile);
                System.out.println("High resolution screenshot saved: " + outputFile.getAbsolutePath());
                System.out.println("Resolution: " + res + "x" + res * screenHratio);
            } catch (Exception e) {
                System.err.println("Failed to save screenshot: " + e.getMessage());
                e.printStackTrace();
            }
        }

        public static String getString(String filePath) throws IOException {
            return new String(Files.readAllBytes(Paths.get(filePath)));
        }

        public static vec getCursorPos(long windowID) {
            DoubleBuffer posX = BufferUtils.createDoubleBuffer(1);
            DoubleBuffer posY = BufferUtils.createDoubleBuffer(1);
            glfwGetCursorPos(windowID, posX, posY);
            return new vec(posX.get(0), posY.get(0), 0);
        }
    }

    //scene
    public class scene {
        public static void addObject(String filepath, int material, vec scale, vec shift, vec rot) {
            Path path = Paths.get(filepath);
            if (Files.isDirectory(path)) {
                File folder = path.toFile();
                File[] mtlFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".mtl"));
                File[] objFiles = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".obj"));
                if (objFiles != null && objFiles.length > 0) {
                    for (File mtlFile : mtlFiles) {
                        parseMtls(mtlFile.getPath(), filepath);
                    }
                    for (File objFile : objFiles) {
                        parseObj(objFile.getPath(), material, scale, shift, rot, filepath);
                    }
                } else {
                    System.out.println("no obj files found in the directory.");
                }
            } else if (Files.isRegularFile(path) && filepath.toLowerCase().endsWith(".obj")) {
                parseObj(filepath, material, scale, shift, rot, null);
            }
        }

        public static void parseObj(String filePath, int material, vec scale, vec shift, vec rot, String parentDirectory) {
            List<Double> currentObjVertsx = new ArrayList<>();
            List<Double> currentObjVertsy = new ArrayList<>();
            List<Double> currentObjVertsz = new ArrayList<>();

            int objectStartTri = triangles.size();

            int mtl = -1;
            int count = 0;
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                List<vec> vertices = new ArrayList<>();
                vertices.add(new vec(0));
                List<vec> normals = new ArrayList<>();
                normals.add(new vec(0));
                List<vec> texture_coordinates = new ArrayList<>();
                texture_coordinates.add(new vec(69.420,0,0));
                while ((line = br.readLine()) != null) {
                    count++;
                    if (line.startsWith("o ") || line.startsWith("g ")) {
                        mtl = material;
                        // new object, add AABB for previous object
                        if (triangles.size() > objectStartTri) {
                            // AABB for current object
                            if (!currentObjVertsx.isEmpty()) {
                                sceneObjs.add(new BVH(objectStartTri, triangles.size()));
                                // Clear current object vertices for next object
                                currentObjVertsx.clear();
                                currentObjVertsy.clear();
                                currentObjVertsz.clear();
                            }
                        }
                        objectStartTri = triangles.size();
                        continue;
                    }
                    if (line.startsWith("usemtl ")) {
                        String name = line.split(" ")[1].trim() + parentDirectory;
                        int i = 0;
                        for (material currMaterial : materials) {
                            if (currMaterial.name.equals(name)) {
                                mtl = i;
                            }
                            i++;
                        }
                    } else if (line.startsWith("v ")) {
                        // add vertices
                        String[] parts = line.split("\\s+");
                        double x = Double.parseDouble(parts[1]);
                        double y = Double.parseDouble(parts[2]);
                        double z = Double.parseDouble(parts[3]);
                        vec vf = (((new vec(x, y, z)).mult(scale)).rotate(rot).add(shift));

                        currentObjVertsx.add(vf.x);
                        currentObjVertsy.add(vf.y);
                        currentObjVertsz.add(vf.z);

                        vertices.add(vf);
                    } else if (line.startsWith("vt ")) {
                        String[] parts = line.split("\\s+");
                        texture_coordinates.add(new vec(Double.parseDouble(parts[1]), Double.parseDouble(parts[2]), 0));
                    } else if (line.startsWith("vn ")) {
                        String[] parts = line.split("\\s+");
                        double x = Double.parseDouble(parts[1]);
                        double y = Double.parseDouble(parts[2]);
                        double z = Double.parseDouble(parts[3]);
                        normals.add((new vec(x, y, z).mult(scale)).rotate(rot));
                    } else if (line.startsWith("f ")) {
                        String[] parts = line.trim().substring(2).split("\\s+");

                        vec vertIndices = new vec(-1, -1, -1);
                        vec textureIndices = new vec(0, 0, 0);
                        vec normalIndices = new vec(0, 0, 0);

                        for (int i = 0; i < 3; i++) {
                            String[] components = parts[i].split("/");

                            if (!components[0].isEmpty()) {
                                if (i == 0) vertIndices.x = Integer.parseInt(components[0]);
                                else if (i == 1) vertIndices.y = Integer.parseInt(components[0]);
                                else vertIndices.z = Integer.parseInt(components[0]);
                            }

                            if (components.length > 1 && !components[1].isEmpty()) {
                                if (i == 0) textureIndices.x = Integer.parseInt(components[1]);
                                else if (i == 1) textureIndices.y = Integer.parseInt(components[1]);
                                else textureIndices.z = Integer.parseInt(components[1]);
                            }

                            if (components.length > 2 && !components[2].isEmpty()) {
                                if (i == 0) normalIndices.x = Integer.parseInt(components[2]);
                                else if (i == 1) normalIndices.y = Integer.parseInt(components[2]);
                                else normalIndices.z = Integer.parseInt(components[2]);
                            }
                        }

                        triangle newTri = new triangle( vertices.get((int) vertIndices.x),vertices.get((int) vertIndices.y),vertices.get((int) vertIndices.z),
                                normals.get((int) normalIndices.x),normals.get((int) normalIndices.y),normals.get((int) normalIndices.z),
                                texture_coordinates.get((int) textureIndices.x),texture_coordinates.get((int) textureIndices.y),texture_coordinates.get((int) textureIndices.z),mtl);
                        triangles.add(newTri);
                    }
                }

                System.out.println("parsed " + filePath);

                // Add AABB for the last object
                if (triangles.size() > objectStartTri) {
                    if (!currentObjVertsx.isEmpty()) {
                        System.out.println("adding BVH");
                        sceneObjs.add(new BVH(objectStartTri, triangles.size()));
                    }
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        public static void addImplicit(int Fn, vec shift, vec scale, vec rot, int m) {
            fn.add(Fn);
            Ishift.add(shift);
            Iscale.add(scale);
            Irot.add(rot);
            Im.add(m);
        }

        public static void addTri(vec v1, vec v2, vec v3, int m) {
            triangles.add(new triangle(v1,v2,v3,new vec(0),new vec(0),new vec(0),new vec(0),new vec(0),new vec(0),m));
        }

        public static void addEllipsoid(vec c, vec stretch, vec rot, float radius, int m) {
            Ec.add(c);
            Estretch.add(stretch);
            Erot.add(rot);
            Erad.add(radius);
            Em.add(m);
        }

        public static void addMat(double KaR, double KaG, double KaB, double KdR, double KdG, double KdB, double KsR, double KsG, double KsB, double alpha, double reflectivity, double transmission, double refIndex, double emission, double emissiveRadius, double solid, double absorption, double textureID) {
            List<Float> mat = new ArrayList<>();
            mat.add((float) KaR);
            mat.add((float) KaG);
            mat.add((float) KaB);
            mat.add((float) KdR);
            mat.add((float) KdG);
            mat.add((float) KdB);
            mat.add((float) KsR);
            mat.add((float) KsG);
            mat.add((float) KsB);
            mat.add((float) alpha);
            mat.add((float) reflectivity);
            mat.add((float) transmission);
            mat.add((float) refIndex);
            mat.add((float) emission);
            mat.add((float) emissiveRadius);
            mat.add((float) solid);
            mat.add((float) absorption);
            mat.add((float) textureID);
            mats.add(mat);
        }

        public static void addMaterial(String name) {
            material mtl = new material();
            mtl.name = name;
            materials.add(mtl);
        }

        public static void setLastMtl(String property, Object val) {
            try {
                Field field = materials.get(materials.size() - 1).getClass().getDeclaredField(property);
                field.setAccessible(true);
                field.set(materials.get(materials.size() - 1), val);
            } catch (NoSuchFieldException | IllegalAccessException e) {
                throw new RuntimeException("Not a valid property");
            }
        }

    }

    //datatype
    public static class vec {
        double x, y, z, w;

        public vec(double x, double y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        public vec(double x, double y) {
            this.x = x;
            this.y = y;
        }
        public vec(double x, double y, double z, double w) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }
        public vec(vec vector, double w) {
            this.x = vector.x;
            this.y = vector.y;
            this.z = vector.z;
            this.w = w;
        }
        public vec(vec vector) {
            x = vector.x;
            y = vector.y;
            z = vector.z;
        }
        public vec(double v) {
            this.x = v;
            this.y = v;
            this.z = v;
        }

        public vec add(vec vector) {
            return new vec(x + vector.x, y + vector.y, z + vector.z);
        }
        public vec sub(vec vector) {
            return new vec(x - vector.x, y - vector.y, z - vector.z);
        }
        public vec mult(vec vector) {
            return new vec(x * vector.x, y * vector.y, z * vector.z);
        }
        public vec mult(double c) {
            return new vec(x * c, y * c, z * c);
        }
        public vec mult(int c) {
            return new vec(x * c, y * c, z * c);
        }
        public vec mult(double x, double y, double z) {
            return new vec(this.x * x, this.y * y, this.z * z);
        }
        public vec div(vec vector) {
            return new vec(x / vector.x, y / vector.y, z / vector.z);
        }
        public vec div(double c) {
            return new vec(x / c, y / c, z / c);
        }

        public static vec mean(List<vec> list) {
            vec sum = new vec(0, 0, 0);
            for (int i = 0; i < list.size(); i++) {
                sum.add(list.get(i));
            }
            return sum.div(list.size());
        }
        public static vec min(List<vec> list) {
            vec min = new vec(Double.MAX_VALUE);
            for (dispatch.vec vec : list) {
                if (vec.x < min.x) min.x = vec.x;
                if (vec.y < min.y) min.y = vec.y;
                if (vec.z < min.z) min.z = vec.z;
            }
            return min;
        }
        public static vec max(List<vec> list) {
            vec max = new vec(Double.NEGATIVE_INFINITY);
            for (vec v : list) {
                if (v.x > max.x) max.x = v.x;
                if (v.y > max.y) max.y = v.y;
                if (v.z > max.z) max.z = v.z;
            }
            return max;
        }

        public double[] toArray() {
            return new double[]{x, y, z};
        }

        public vec rotate(vec rot) {
            double rx = rot.x;
            double ry = rot.y;
            double rz = rot.z;
            // Convert angles from degrees to radians
            double radX = rx;
            double radY = ry;
            double radZ = rz;

            // Rotation around the X-axis
            double cosX = Math.cos(radX);
            double sinX = Math.sin(radX);
            double newY = cosX * y - sinX * z;
            double newZ = sinX * y + cosX * z;
            y = newY;
            z = newZ;

            // Rotation around the Y-axis
            double cosY = Math.cos(radY);
            double sinY = Math.sin(radY);
            double newX = cosY * x + sinY * z;
            newZ = -sinY * x + cosY * z;
            x = newX;
            z = newZ;

            // Rotation around the Z-axis
            double cosZ = Math.cos(radZ);
            double sinZ = Math.sin(radZ);
            newX = cosZ * x - sinZ * y;
            newY = sinZ * x + cosZ * y;
            x = newX;
            y = newY;

            return new vec(newX, newY, newZ);
        }

        public double dot(vec vector) {
            return x * vector.x + y * vector.y + z * vector.z;
        }

        public vec cross(vec vector) {
            return new vec(((y * vector.z) - (z * vector.y)), ((z * vector.x) - (x * vector.z)), ((x * vector.y) - (y * vector.x)));
        }

        public double magnitude() {
            return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
        }

        public vec normalize() {
            double mag = this.magnitude();
            return new vec(this.x / mag, this.y / mag, this.z / mag);
        }

        public double dist(vec vector) {
            return Math.sqrt(Math.pow(x - vector.x, 2) + Math.pow(y - vector.y, 2) + Math.pow(z - vector.z, 2));
        }

        public void println() {
            System.out.print("(" + (x) + "," + (y) + "," + (z) + ")\n");
        }
    }


    public static class triangle {
        vec v1;
        vec v2;
        vec v3;
        vec n1;
        vec n2;
        vec n3;
        vec vt1;
        vec vt2;
        vec vt3;
        int material;
        int ID;

        vec min;
        vec max;
        vec centroid;

        public triangle(vec v1, vec v2, vec v3, vec n1, vec n2, vec n3, vec vt1, vec vt2, vec vt3, int material) {
            this.v1 = v1;
            this.v2 = v2;
            this.v3 = v3;
            this.n1 = n1.normalize();
            this.n2 = n2.normalize();
            this.n3 = n3.normalize();
            this.vt1 = vt1;
            this.vt2 = vt2;
            this.vt3 = vt3;
            this.material = material;
            List<vec> verts = new ArrayList<>();
            verts.add(this.v1); verts.add(this.v2); verts.add(this.v3);
            this.min = vec.min(verts);
            this.max = vec.max(verts);
            this.centroid = (v1.add(v2.add(v3))).div(3);
            this.ID = NEXT_TRI_ID;
            NEXT_TRI_ID++;
        }

        public triangle(triangle tri) {
            this.v1 = new vec(tri.v1);
            this.v2 = new vec(tri.v2);
            this.v3 = new vec(tri.v3);
            this.n1 = new vec(tri.n1);
            this.n2 = new vec(tri.n2);
            this.n3 = new vec(tri.n3);
            this.vt1 = new vec(tri.vt1);
            this.vt2 = new vec(tri.vt2);
            this.vt3 = new vec(tri.vt3);
            this.material = tri.material;
            this.ID = tri.ID;
            this.min = new vec(tri.min);
            this.max = new vec(tri.max);
            this.centroid = new vec(tri.centroid);
        }

        public int ID() {
            return this.ID;
        }
    }

    public static class material {
        String name;
        vec Ka;
        vec Kd;
        vec Ks;
        double Ns; // specular exponent
        double d; // dissolved (transparency 1-0, 1 is opaque)
        double Tr; // occasionally used, opposite of d (0 is opaque)
        vec Tf; // transmission filter
        double Ni; // refractive index
        vec Ke; // emission color
        int illum; // shading model (0-10, each has diff properties)
        int map_Ka;
        int map_Kd;
        int map_Ks;
        //PBR extension types
        double Pm; // metallicity (0-1, dielectric to metallic)
        double Pr; // roughness (0-1, perfectly smooth to "extremely" rough)
        double Ps; // sheen (0-1, no sheen effect to maximum sheen)
        double Pc; // clearcoat thickness (0-1, smooth clearcoat to rough clearcoat (blurry reflections))
        double Pcr;
        double aniso; // anisotropy (0-1, isotropic surface to fully anisotropic) (uniform-directional reflections)
        double anisor; // rotational anisotropy (0-1, but essentially 0-2pi, rotates pattern of anisotropy)
        int map_Pm;
        int map_Pr;
        int map_Ps;
        int map_Pc;
        int map_Pcr;
        int map_bump;
        int map_d;
        int map_Tr;
        int map_Ns;
        int map_Ke;
        //custom
        double Density;
        double subsurface;
        vec subsurfaceColor;
        vec subsurfaceRadius;


        public static void parseMtls(String filePath, String parentDirectoryPath) {
            System.out.println("uploading textures and materials...");
            long start = System.nanoTime();
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                while ((line = br.readLine()) != null) {
                    if (line.startsWith("newmtl ")) {
                        String[] parts = line.split(" ");
                        material mat = new material();
                        mat.name = parts[1].trim() + parentDirectoryPath;
                        while ((line = br.readLine()) != null && !line.equals("")) {
                            line = line.replace("/","\\");
                            if (line.startsWith("Ka ")) {
                                String[] vals = line.split(" ");
                                mat.Ka = new vec(Double.parseDouble(vals[1].trim()), Double.parseDouble(vals[2].trim()), Double.parseDouble(vals[3].trim()));
                            } else if (line.startsWith("Kd ")) {
                                String[] vals = line.split(" ");
                                mat.Kd = new vec(Double.parseDouble(vals[1].trim()), Double.parseDouble(vals[2].trim()), Double.parseDouble(vals[3].trim()));
                            } else if (line.startsWith("Ks ")) {
                                String[] vals = line.split(" ");
                                mat.Ks = new vec(Double.parseDouble(vals[1].trim()), Double.parseDouble(vals[2].trim()), Double.parseDouble(vals[3].trim()));
                            } else if (line.startsWith("Ns ")) {
                                String[] vals = line.split(" ");
                                mat.Ns = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("d ")) {
                                String[] vals = line.split(" ");
                                mat.d = Double.parseDouble(vals[1].trim());
                                mat.Tr = 1 - mat.d;
                            } else if (line.startsWith("Tr ")) {
                                String[] vals = line.split(" ");
                                mat.Tr = Double.parseDouble(vals[1].trim());
                                mat.d = 1 - mat.Tr;
                            } else if (line.startsWith("Tf ")) {
                                String[] vals = line.split(" ");
                                mat.Tf = new vec(Double.parseDouble(vals[1].trim()), Double.parseDouble(vals[2].trim()), Double.parseDouble(vals[3].trim()));
                            } else if (line.startsWith("Ni ")) {
                                String[] vals = line.split(" ");
                                mat.Ni = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("Ke ")) {
                                String[] vals = line.split(" ");
                                vec Ke = new vec(Double.parseDouble(vals[1].trim()), Double.parseDouble(vals[2].trim()), Double.parseDouble(vals[3].trim()));
                                mat.Ke = Ke;
                                mat.Density = Ke.magnitude();
                            } else if (line.startsWith("Density ")) {
                                String[] vals = line.split(" ");
                                mat.Density = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("illum ")) {
                                String[] vals = line.split(" ");
                                mat.illum = Integer.parseInt(vals[1].trim());
                            } else if (line.startsWith("map_Ka ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Ka = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Ka = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Kd ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Kd = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Kd = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Ks ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Ks = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Ks = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("Pm ")) {
                                String[] vals = line.split(" ");
                                mat.Pm = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("Pr ")) {
                                String[] vals = line.split(" ");
                                mat.Pr = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("Ps ")) {
                                String[] vals = line.split(" ");
                                mat.Ps = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("Pc ")) {
                                String[] vals = line.split(" ");
                                mat.Pc = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("Pcr ")) {
                                String[] vals = line.split(" ");
                                mat.Pcr = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("aniso ")) {
                                String[] vals = line.split(" ");
                                mat.aniso = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("anisor ")) {
                                String[] vals = line.split(" ");
                                mat.anisor = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("map_Pm ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Pm = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Pm = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Pr ") || line.startsWith("refl")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Pr = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Pr = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Ps ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Ps = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Ps = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Pc ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Pc = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Pc = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Pcr ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Pcr = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Pcr = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Bump ") || line.startsWith("bump ") || line.startsWith("map_bump ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_bump = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_bump = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_d ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_d = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_d = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Tr ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Tr = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Tr = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Ns ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Ns = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Ns = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } else if (line.startsWith("map_Ke ")) {
                                String[] vals = line.split(" ");
                                if (textureNames.contains(vals[1].trim())) {
                                    mat.map_Ke = textureNames.indexOf(vals[1].trim());
                                } else {
                                    mat.map_Ke = textures.size();
                                    parseTexture(parentDirectoryPath + "\\" + (vals[1].trim()), (vals[1].trim()));
                                }
                            } /* Custom */ else if (line.startsWith("subsurface ")) {
                                String[] vals = line.split(" ");
                                mat.subsurface = Double.parseDouble(vals[1].trim());
                            } else if (line.startsWith("subsurfaceColor ")) {
                                String[] vals = line.split(" ");
                                mat.subsurfaceColor = new vec(Double.parseDouble(vals[1].trim()), Double.parseDouble(vals[2].trim()), Double.parseDouble(vals[3].trim()));
                            } else if (line.startsWith("subsurfaceRadius ")) {
                                String[] vals = line.split(" ");
                                mat.subsurfaceRadius = new vec(Double.parseDouble(vals[1].trim()), Double.parseDouble(vals[2].trim()), Double.parseDouble(vals[3].trim()));
                            }
                        }
                        materials.add(mat);
                    }
                }
                System.out.println();
                System.out.println("    (took " + ((System.nanoTime() - start) / 1000000000.0) + " seconds)");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        public material() {
            this.Ka = new vec(0);
            this.Kd = new vec(0.8);
            this.Ks = new vec(0.5);
            this.Ns = 10;
            this.d = 0;
            this.Tr = 0;
            this.Tf = new vec(0);
            this.Ni = 1;
            this.Ke = new vec(0);
            this.Density = 1;
            this.map_Ka = -1;
            this.map_Kd = -1;
            this.map_Ks = -1;
            this.map_Ke = -1;
            this.illum = 0;
            this.Pm = 0;
            this.Pr = 1;
            this.Ps = 0;
            this.Pc = 0;
            this.Pcr = 0;
            this.aniso = 0;
            this.anisor = 0;
            this.map_Pm = -1;
            this.map_Pr = -1;
            this.map_Ps = -1;
            this.map_Pc = -1;
            this.map_Pcr = -1;
            this.map_bump = -1;
            this.map_d = -1;
            this.map_Tr = -1;
            this.map_Ns = -1;
            //custom
            this.subsurface = 0;
            this.subsurfaceColor = new vec(0);
            this.subsurfaceRadius = new vec(0);
        }

        public static void parseTexture(String filePath, String name) {
            System.out.print("\r    parsing (" + filePath + ")...");
            BufferedImage image = null;
            try {
                image = ImageIO.read(new File(filePath));
                int width = image.getWidth();
                int height = image.getHeight();
                ByteBuffer buffer = BufferUtils.createByteBuffer(width * height * 4);
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        int pixel = image.getRGB(x, y);
                        buffer.put((byte) ((pixel >> 16) & 0xFF)); // Red
                        buffer.put((byte) ((pixel >> 8) & 0xFF));  // Green
                        buffer.put((byte) (pixel & 0xFF));         // Blue
                        buffer.put((byte) ((pixel >> 24) & 0xFF)); // Alpha
                    }
                }
                buffer.flip();
                textureNames.add(name);
                textures.add(filePath);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    //shader util
    public static class BVH {
        vec min;
        vec max;
        List<Integer> storedTri;
        BVH LeftChildBVH;
        BVH RightChildBVH;
        int branchDepth;
        int ID;

        public static class BoundingBox {
            vec Min;
            vec Max;
            vec Center;
            vec Size;
            boolean hasPoint;

            public BoundingBox() {
                this.Min = null;
                this.Max = null;
                this.Center = null;
                this.Size = null;
                this.hasPoint = false;
            }
            public BoundingBox(BoundingBox duplicate) {
                this.Min = duplicate.Min;
                this.Max = duplicate.Max;
                this.Center = duplicate.Center;
                this.Size = duplicate.Size;
                this.hasPoint = duplicate.hasPoint;
            }
            public void GrowToInclude(triangle tri) {
                GrowToInclude(new vec(tri.min), new vec(tri.max));
            }
            public void GrowToInclude(vec min, vec max) {
                if (hasPoint) {
                    this.Min.x = Math.min(min.x, this.Min.x);
                    this.Min.y = Math.min(min.y, this.Min.y);
                    this.Min.z = Math.min(min.z, this.Min.z);
                    this.Max.x = Math.max(max.x, this.Max.x);
                    this.Max.y = Math.max(max.y, this.Max.y);
                    this.Max.z = Math.max(max.z, this.Max.z);
                } else {
                    this.hasPoint = true;
                    this.Min = min;
                    this.Max = max;
                }
                this.Center = (this.Min.add(this.Max)).div(2);
                this.Size = this.Max.sub(this.Min);
            }
        }

        public BVH(int triIndicesStart, int triIndicesEnd) {
            this.ID = nextBVHId;
            nextBVHId++;
            List<triangle> tris = new ArrayList<>();
            this.branchDepth = 0;
            BoundingBox thisBounds = new BoundingBox();
            for (int i = triIndicesStart; i < triIndicesEnd; i++) {
                tris.add(new triangle(triangles.get(i)));
                thisBounds.GrowToInclude(new vec(triangles.get(i).min), new vec(triangles.get(i).max));
            }
            this.min = new vec(thisBounds.Min);
            this.max = new vec(thisBounds.Max);
            this.storedTri = tris.stream().map(triangle::ID).collect(Collectors.toCollection(ArrayList::new));
            List<BVH> children = splitTEST(thisBounds, tris, Double.POSITIVE_INFINITY, 0);
            this.LeftChildBVH = children.get(0);
            this.RightChildBVH = children.get(1);
        }
        public List<BVH> splitTEST(BoundingBox bounds, List<triangle> tris, double bestCost, int branchDepth) {
            List<BVH> children = new ArrayList<>();

            List<triangle> leftTris = new ArrayList<>(), rightTris = new ArrayList<>();
            List<Integer> leftIDs = new ArrayList<>(), rightIDs = new ArrayList<>();
            BoundingBox leftBounds = new BoundingBox();
            BoundingBox rightBounds = new BoundingBox();

            int bestAxis = 0;
            double bestPos = -1;

            for (int axis = 0; axis < 3; axis++) {
                for (int i = 0; i < OPTIMIZATION_LEVEL; i++) {
                    double splitPercent = ((double)i+1.0)/(OPTIMIZATION_LEVEL+1.0);
                    double pos = (bounds.Min.add(bounds.Size.mult(splitPercent))).toArray()[axis];
                    double cost = testSplitOnTEST(axis, pos, new ArrayList<>(tris));
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestAxis = axis;
                        bestPos = pos;
                    }
                }
            }
            if (bestPos == -1) return children;

            for (triangle tri : tris) {
                if (tri.centroid.toArray()[bestAxis] < bestPos) {
                    leftTris.add(tri);
                    leftBounds.GrowToInclude(tri.min, new vec(tri.max));
                    leftIDs.add(tri.ID);
                } else {
                    rightTris.add(tri);
                    rightBounds.GrowToInclude(tri.min,new vec(tri.max));
                    rightIDs.add(tri.ID);
                }
            }


            if (leftTris.isEmpty() || leftTris.size() == tris.size()) {
                children.add(null);
            } else {
                BVH leftChild = new BVH(leftBounds.Min, leftBounds.Max, leftIDs, branchDepth + 1);
                if (branchDepth >= MAX_BVH_BRANCHES || leftTris.size() <= MAX_TRIS_IN_BVH_LEAF) {
                    children.add(leftChild);
                } else {
                    List<BVH> Lchildren = splitTEST(new BoundingBox(leftBounds), new ArrayList<>(leftTris), bestCost, branchDepth+1);
                    if (Lchildren.isEmpty()) {
                        children.add(leftChild);
                    } else {
                        leftChild.LeftChildBVH = Lchildren.get(0);
                        leftChild.RightChildBVH = Lchildren.get(1);
                        children.add(leftChild);
                    }
                }
            }

            if (rightTris.isEmpty() || rightTris.size() == tris.size()) {
                children.add(null);
            } else {
                BVH rightChild = new BVH(rightBounds.Min, rightBounds.Max, rightIDs, branchDepth + 1);
                if (branchDepth >= MAX_BVH_BRANCHES || rightTris.size() <= MAX_TRIS_IN_BVH_LEAF) {
                    children.add(rightChild);
                } else {
                    List<BVH> Rchildren = splitTEST(new BoundingBox(rightBounds), new ArrayList<>(rightTris), bestCost, branchDepth+1);
                    if (Rchildren.isEmpty()) {
                        children.add(rightChild);
                    } else {
                        rightChild.LeftChildBVH = Rchildren.get(0);
                        rightChild.RightChildBVH = Rchildren.get(1);
                        children.add(rightChild);
                    }
                }
            }
            return children;
        }
        private double testSplitOnTEST(int axis, double pos, List<triangle> tris) {
            int numLeft = 0;
            int numRight = 0;
            BoundingBox leftBounds = new BoundingBox(), rightBounds = new BoundingBox();
            for (triangle tri : tris) {
                double[] centroid = tri.centroid.toArray();

                List<vec> verts = new ArrayList<>();
                verts.add(tri.v1);
                verts.add(tri.v2);
                verts.add(tri.v3);
                //if (tri.min.sub(vec.min(verts)).magnitude() != 0) System.out.println("min is wrong in testSplitOnTEST" + tri.min.sub(vec.min(verts)).magnitude());
                tri.min = vec.min(verts);

                if (centroid[axis] < pos) {
                    leftBounds.GrowToInclude(tri);
                    numLeft++;
                } else {
                    rightBounds.GrowToInclude(tri);
                    numRight++;
                }
            }
            double Lcost = (numLeft == 0 ? Double.POSITIVE_INFINITY : cost(leftBounds.Size, numLeft));
            double Rcost = (numRight == 0 ? Double.POSITIVE_INFINITY : cost(rightBounds.Size, numRight));
            return Lcost + Rcost;
        }
        public double cost(vec extent, int numTri) {
            if (numTri == 0) return Double.POSITIVE_INFINITY;
            double halfSurfaceArea = (extent.x * extent.y + extent.x * extent.z + extent.y * extent.z);
            return Math.abs(halfSurfaceArea) * (double)numTri;
        }


        public BVH(vec min, vec max, List<Integer> tris, int branchDepth) {
            this.ID = nextBVHId;
            nextBVHId++;
            this.min = min;
            this.max = max;
            this.branchDepth = branchDepth;
            this.storedTri = tris;
        }

        public static void allBVHtoList() {
            //clear global lists.
            BVHleafTriIndices.clear();
            sceneBVHdata.clear();
            sceneBVHtree.clear();
            objIndicesInTree.clear();
            for (int i = 0; i < nextBVHId * 8; i++) {
                sceneBVHdata.add(0.0f);
            }
            int counter = 0;
            for (BVH currentBVH : sceneObjs) {
                counter++;
                if (currentBVH != null) {
                    objIndicesInTree.add(currentBVH.ID);
                    flattenBVH(currentBVH, sceneBVHdata, sceneBVHtree);
                    System.out.print("\r    created (" + counter + "/" + sceneObjs.size() + ") BVH");
                }
            }
            System.out.println();
            //post process data lists
            sortTree();
        }
        private static void flattenBVH(BVH node, List<Float> data, List<Integer> tree) {
            if (node == null) return;
            if (node.LeftChildBVH == null && node.RightChildBVH == null) {
                // Store starting index before adding triangles
                int startIdx = BVHleafTriIndices.size();
                // Add all triangles from this leaf
                BVHleafTriIndices.addAll(node.storedTri);
                // Store the index range
                data.set(8 * node.ID + 6, (float) startIdx);
                data.set(8 * node.ID + 7, (float) BVHleafTriIndices.size());
            }

//im attemping to raytrace a 4000000 tri scene rn


            data.set(8 * node.ID, (float) node.min.x);
            data.set(8 * node.ID + 1, (float) node.min.y);
            data.set(8 * node.ID + 2, (float) node.min.z);
            data.set(8 * node.ID + 3, (float) node.max.x);
            data.set(8 * node.ID + 4, (float) node.max.y);
            data.set(8 * node.ID + 5, (float) node.max.z);

            int leftChildId = (node.LeftChildBVH != null) ? node.LeftChildBVH.ID : -1;
            int rightChildId = (node.RightChildBVH != null) ? node.RightChildBVH.ID : -1;
            tree.add(node.ID);
            tree.add(leftChildId);
            tree.add(rightChildId);

            flattenBVH(node.LeftChildBVH, data, tree);
            flattenBVH(node.RightChildBVH, data, tree);
        }
        public static void sortTree() {
            if (sceneBVHtree.size() % 3 != 0) {
                throw new IllegalArgumentException("tree incorrectly generated set");
            }
            ArrayList<ArrayList<Integer>> sets = new ArrayList<>();
            for (int i = 0; i < sceneBVHtree.size(); i += 3) {
                ArrayList<Integer> set = new ArrayList<>(sceneBVHtree.subList(i, i + 3));
                sets.add(set);
            }
            // Sort the sets by their first element (ID)
            sets.sort(Comparator.comparingInt(set -> set.get(0)));
            // Clear the original list and repopulate it with the sorted sets
            sceneBVHtree.clear();
            for (ArrayList<Integer> set : sets) {
                sceneBVHtree.addAll(set);
            }
        }

        public double mean(List<Double> list) {
            double sum = 0;
            for (int i = 0; i < list.size(); i++) {
                sum += list.get(i);
            }
            return sum / list.size();
        }
    }

    private static int createShader(int type, String source) {
        int shader = glCreateShader(type);
        glShaderSource(shader, source);
        glCompileShader(shader);
        checkShaderError(shader, type == GL_COMPUTE_SHADER ? "compute" :
                type == GL_VERTEX_SHADER ? "vertex" : "fragment");
        return shader;
    }

    private static void checkShaderError(int shader, String type) {
        if (glGetShaderi(shader, GL_COMPILE_STATUS) == GL_FALSE) {
            String log = glGetShaderInfoLog(shader);
            throw new RuntimeException(type + " shader compilation failed: " + log);
        }
    }

    private static void checkProgramError(int program) {
        if (glGetProgrami(program, GL_LINK_STATUS) == GL_FALSE) {
            String log = glGetProgramInfoLog(program);
            throw new RuntimeException("Program linking failed: " + log);
        }
    }
}