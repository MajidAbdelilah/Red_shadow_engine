
#include <LLGL-C/LLGL.h>
#include <LLGL-C/LLGLWrapper.h>
#include <LLGL-C/RenderSystem.h>
#include <LLGL-C/SwapChain.h>
#include <LLGL-C/Types.h>
#include <LLGL-C/Window.h>
#include <stdint.h>
#include <stdio.h>
#include <wchar.h>
#include <math.h>
#include "stb/stb_image.h"


#define ARRAY_SIZE(A)   (sizeof(A)/sizeof((A)[0]))
#define MATH_PI         ( 3.141592654f )
#define DEG2RAD(X)      ( (X) * MATH_PI / 180.0f )

typedef struct vertex
{
    float position[3];
    float normal[3];
    float texCoord[2];
}t_vertex;

typedef struct SceneConstants {
  float wvpMatrix[4][4];
  float wMatrix[4][4];
} SceneConstants;


t_vertex plane[4] = (t_vertex[]){
        { { -1, -1, -1 }, {  0,  0, -1 }, { 0, 1 } },
        { { -1,  1, -1 }, {  0,  0, -1 }, { 0, 0 } },
        { {  1,  1, -1 }, {  0,  0, -1 }, { 1, 0 } },
        { {  1, -1, -1 }, {  0,  0, -1 }, { 1, 1 } }
};


static const uint32_t planeIndices[6] = {0,  1,  2,  0,  2,  3};

typedef struct renderer
{
    const char* rendererModule;
    int width;
    int height;
    int samples;
    bool vsync;
    bool debugger;
    wchar_t *windowTitle;
}t_render;

const LLGLClearValue g_defaultClear =
{
    .color = { 0.1f, 0.1f, 0.2f, 1.0f },
    .depth = 1.0f
};
int                     g_renderer          = 0;
LLGLSwapChain           g_swapChain         = LLGL_NULL_OBJECT;
LLGLSurface             g_surface           = LLGL_NULL_OBJECT;
LLGLCommandBuffer       g_commandBuffer     = LLGL_NULL_OBJECT;
LLGLCommandQueue        g_commandQueue      = LLGL_NULL_OBJECT;
LLGLViewport            g_viewport;
float                   g_projection[4][4]  = { { 1.0f, 0.0f, 0.0f, 0.0f },
                                                { 0.0f, 1.0f, 0.0f, 0.0f },
                                                { 0.0f, 0.0f, 1.0f, 0.0f },
                                                { 0.0f, 0.0f, 0.0f, 1.0f } };


static struct ExampleEventStatus
{
    float   mouseMotion[2];
    bool    keyDown[256];
}
g_EventStauts =
{
    .mouseMotion = { 0.0f, 0.0f }
};

const LLGLSamplerDescriptor g_defaultSamplerDesc =
{
    .addressModeU   = LLGLSamplerAddressModeRepeat,
    .addressModeV   = LLGLSamplerAddressModeRepeat,
    .addressModeW   = LLGLSamplerAddressModeRepeat,
    .minFilter      = LLGLSamplerFilterLinear,
    .magFilter      = LLGLSamplerFilterLinear,
    .mipMapFilter   = LLGLSamplerFilterLinear,
    .mipMapEnabled  = true,
    .mipMapLODBias  = 0.0f,
    .minLOD         = 0.0f,
    .maxLOD         = 1000.0f,
    .maxAnisotropy  = 1,
    .compareEnabled = false,
    .compareOp      = LLGLCompareOpLess,
    .borderColor    = { 0.0f, 0.0f, 0.0f, 0.0f },
};


void onKeyDown(LLGLWindow sender, LLGLKey keycode)
{
    g_EventStauts.keyDown[keycode] = true;
}

void onKeyUp(LLGLWindow sender, LLGLKey keycode)
{
    g_EventStauts.keyDown[keycode] = false;
}

void onMouseMotion(LLGLWindow sender, const LLGLOffset2D *motion)
{
    g_EventStauts.mouseMotion[0] = (float)motion->x;
    g_EventStauts.mouseMotion[1] = (float)motion->y;
}


static float aspect_ratio()
{
    LLGLExtent2D swapChainResolution;
    llglGetSurfaceContentSize(g_surface, &swapChainResolution);
    return (float)swapChainResolution.width / (float)swapChainResolution.height;
}

void update_viewport()
{
    LLGLExtent2D swapChainResolution;
    llglGetSurfaceContentSize(g_surface, &swapChainResolution);

    g_viewport.x        = 0.0f;
    g_viewport.y        = 0.0f;
    g_viewport.width    = (float)swapChainResolution.width;
    g_viewport.height   = (float)swapChainResolution.height;
    g_viewport.minDepth = 0.0f;
    g_viewport.maxDepth = 1.0f;
}


static void build_perspective_projection(float m[4][4], float aspect, float nearPlane, float farPlane, float fov, bool isUnitCube)
{
    const float h = 1.0f / tanf(fov * 0.5f);
    const float w = h / aspect;

    m[0][0] = w;
    m[0][1] = 0.0f;
    m[0][2] = 0.0f;
    m[0][3] = 0.0f;

    m[1][0] = 0.0f;
    m[1][1] = h;
    m[1][2] = 0.0f;
    m[1][3] = 0.0f;

    m[2][0] = 0.0f;
    m[2][1] = 0.0f;
    m[2][2] = (isUnitCube ? (farPlane + nearPlane)/(farPlane - nearPlane) : farPlane/(farPlane - nearPlane));
    m[2][3] = 1.0f;

    m[3][0] = 0.0f;
    m[3][1] = 0.0f;
    m[3][2] = (isUnitCube ? -(2.0f*farPlane*nearPlane)/(farPlane - nearPlane) : -(farPlane*nearPlane)/(farPlane - nearPlane));
    m[3][3] = 0.0f;
}


void perspective_projection(float outProjection[4][4], float aspectRatio, float nearPlane, float farPlane, float fieldOfView)
{
    const int rendererID = llglGetRendererID();
    if (rendererID == LLGL_RENDERERID_OPENGL || rendererID == LLGL_RENDERERID_VULKAN)
        build_perspective_projection(outProjection, aspectRatio, nearPlane, farPlane, fieldOfView, /*isUnitCube:*/ true);
    else
        build_perspective_projection(outProjection, aspectRatio, nearPlane, farPlane, fieldOfView, /*isUnitCube:*/ false);
}

int init()
{
    t_render render = {.rendererModule = "OpenGL", .width = 640, .height = 400, .samples = 4, .debugger = true, .vsync = true, .windowTitle = L"red_shadow"};

    if (llglLoadRenderSystem(render.rendererModule) == 0)
    {
        fprintf(stderr, "Failed to load render system: %s\n", "OpenGL");
        return 1;
    }
     LLGLSwapChainDescriptor swapChainDesc =
    {
        .resolution     = { render.width, render.height },
        .colorBits      = 32,   // 32 bits for color information
        .depthBits      = 24,   // 24 bits for depth comparison
        .stencilBits    = 8,    // 8 bits for stencil patterns
        .samples        = render.samples, // check if LLGL adapts sample count that is too high
    };
    g_swapChain = llglCreateSwapChain(&swapChainDesc);
    llglSetVsyncInterval(g_swapChain, 1);

    g_surface = llglGetSurface(g_swapChain);
    LLGLWindow window = LLGL_GET_AS(LLGLWindow, g_surface);
    llglSetWindowTitle(window, render.windowTitle);
    LLGLWindowEventListener windowEventListner = 
    {
        .onKeyDown = onKeyDown,
        .onKeyUp = onKeyUp,
        .onGlobalMotion = onMouseMotion
    };
    llglAddWindowEventListener(window, &windowEventListner);

    llglShowWindow(window, true);

    LLGLCommandBufferDescriptor cmdBufferDesc = 
    {
        .flags = LLGLCommandBufferImmediateSubmit,
        .numNativeBuffers = 2
    };
    g_commandBuffer = llglCreateCommandBuffer(&cmdBufferDesc);

    update_viewport();

    const float aspectRatio = aspect_ratio();
    perspective_projection(g_projection, aspectRatio, /*nearPlane:*/ 0.1f, /*farPlane;*/ 100.0f, /*fieldOfView:*/ DEG2RAD(45.0f));

    return 0;
}

static void reset_event_status()
{
    g_EventStauts.mouseMotion[0] = 0.0f;
    g_EventStauts.mouseMotion[1] = 0.0f;
}


bool example_poll_events()
{
    // Reset event status
    reset_event_status();

    // Process surface and events and check if window was closed
    return llglProcessSurfaceEvents(g_surface) && !llglHasWindowQuit(LLGL_GET_AS(LLGLWindow, g_surface)) && !g_EventStauts.keyDown[LLGLKeyEscape];
}
bool key_pressed(LLGLKey keyCode)
{
    return g_EventStauts.keyDown[keyCode];
}

float mouse_movement_x()
{
    return g_EventStauts.mouseMotion[0];
}

float mouse_movement_y()
{
    return g_EventStauts.mouseMotion[1];
}


#define foreach_matrix_element(R, C) \
    for (int R = 0; R < 4; ++R) for (int C = 0; C < 4; ++C)

void matrix_load_identity(float outMatrix[4][4])
{
    foreach_matrix_element(r, c)
        outMatrix[c][r] = (r == c ? 1.0f : 0.0f);
}

void matrix_mul(float outMatrix[4][4], const float inMatrixLhs[4][4], const float inMatrixRhs[4][4])
{
    foreach_matrix_element(r, c)
    {
        outMatrix[c][r] = 0.0f;
        for (int i = 0; i < 4; ++i)
            outMatrix[c][r] += inMatrixLhs[i][r] * inMatrixRhs[c][i];
    }
}

void matrix_translate(float outMatrix[4][4], float x, float y, float z)
{
    outMatrix[3][0] += outMatrix[0][0]*x + outMatrix[1][0]*y + outMatrix[2][0]*z;
    outMatrix[3][1] += outMatrix[0][1]*x + outMatrix[1][1]*y + outMatrix[2][1]*z;
    outMatrix[3][2] += outMatrix[0][2]*x + outMatrix[1][2]*y + outMatrix[2][2]*z;
}

void matrix_rotate(float outMatrix[4][4], float x, float y, float z, float angle)
{
    const float c  = cosf(angle);
    const float s  = sinf(angle);
    const float cc = 1.0f - c;

    const float vecInvLen = 1.0f / sqrtf(x*x + y*y + z*z);
    x *= vecInvLen;
    y *= vecInvLen;
    z *= vecInvLen;

    outMatrix[0][0] = x*x*cc + c;
    outMatrix[0][1] = x*y*cc - z*s;
    outMatrix[0][2] = x*z*cc + y*s;

    outMatrix[1][0] = y*x*cc + z*s;
    outMatrix[1][1] = y*y*cc + c;
    outMatrix[1][2] = y*z*cc - x*s;

    outMatrix[2][0] = x*z*cc - y*s;
    outMatrix[2][1] = y*z*cc + x*s;
    outMatrix[2][2] = z*z*cc + c;
}

void releaseSystem()
{
    llglUnloadRenderSystem();
}


int main()
{
    if (init(L"Texturing") != 0)
        return 1;
    const t_vertex vertices[] = {plane[0], plane[1], plane[2], plane[3]};
    size_t vertexCount = 4;
    const uint32_t indices[] = {planeIndices[0], planeIndices[1], planeIndices[2], planeIndices[3], planeIndices[4], planeIndices[5]};
    size_t indexCount = 6;

     // Vertex format with 3D position, normal, and texture-coordinates
      const LLGLVertexAttribute vertexAttributes[3] = {
      {.name = "position",
       .format = LLGLFormatRGB32Float,
       .location = 0,
       .offset = offsetof(t_vertex, position),
       .stride = sizeof(t_vertex)},
      {.name = "normal",
       .format = LLGLFormatRGB32Float,
       .location = 1,
       .offset = offsetof(t_vertex, normal),
       .stride = sizeof(t_vertex)},
      {.name = "texCoord",
       .format = LLGLFormatRG32Float,
       .location = 2,
       .offset = offsetof(t_vertex, texCoord),
       .stride = sizeof(t_vertex)},
  };

      // Create vertex buffer
  const LLGLBufferDescriptor vertexBufferDesc = {
      .size = sizeof(t_vertex) *
              vertexCount,               // Size (in bytes) of the vertex buffer
      .bindFlags = LLGLBindVertexBuffer, // Enables the buffer to be bound to a
                                         // vertex buffer slot
      .numVertexAttribs = ARRAY_SIZE(vertexAttributes),
      .vertexAttribs = vertexAttributes, // Vertex format layout
  };
    LLGLBuffer vertexBuffer = llglCreateBuffer(&vertexBufferDesc, vertices);

    // Create index buffer
  const LLGLBufferDescriptor indexBufferDesc = {
      .size =
          sizeof(uint32_t) * indexCount, // Size (in bytes) of the index buffer
      .bindFlags = LLGLBindIndexBuffer,  // Enables the buffer to be bound to an
                                         // index buffer slot
  };
  LLGLBuffer indexBuffer = llglCreateBuffer(&indexBufferDesc, indices);

  // Create constant buffer
  const LLGLBufferDescriptor sceneBufferDesc = {
      .size = sizeof(SceneConstants), // Size (in bytes) of the constant buffer
      .bindFlags =
          LLGLBindConstantBuffer, // Enables the buffer to be bound as a
                                  // constant buffer, which is optimized for
                                  // fast updates per draw call
  };
  LLGLBuffer sceneBuffer = llglCreateBuffer(&sceneBufferDesc, NULL);

      // Load image data from file (using STBI library, see
  // http://nothings.org/stb_image.h)
  const char *imageFilename = "./Media/Textures/Crate.jpg";

  int imageSize[2] = {0, 0}, texComponents = 0;
  unsigned char *imageBuffer = stbi_load(imageFilename, &imageSize[0], &imageSize[1], &texComponents, 0);
  if (!imageBuffer) {
    fprintf(stderr, "Failed to load image: %s\n", imageFilename);
    return 1;
  }

      // Create texture
  const LLGLImageView imageView = {
      .format = (texComponents == 4
                     ? LLGLImageFormatRGBA
                     : LLGLImageFormatRGB), // Image color format (RGBA or RGB)
      .dataType = LLGLDataTypeUInt8,        // Data tpye (unsigned char => 8-bit
                                            // unsigned integer)
      .data = imageBuffer,                  // Image source buffer
      .dataSize = (size_t)(imageSize[0] * imageSize[1] *
                           texComponents), // Image buffer size
  };

      const LLGLTextureDescriptor texDesc = {
      .type = LLGLTextureTypeTexture2D,
      .format = LLGLFormatRGBA8UNorm,
      .extent = {(uint32_t)imageSize[0], (uint32_t)imageSize[1], 1u},
      .miscFlags = LLGLMiscGenerateMips,
  };
  LLGLTexture colorTexture = llglCreateTexture(&texDesc, &imageView);

    // Create samplers
  LLGLSampler samplers[3];

  LLGLSamplerDescriptor anisotropySamplerDesc = g_defaultSamplerDesc;
  { anisotropySamplerDesc.maxAnisotropy = 8; }
  samplers[0] = llglCreateSampler(&anisotropySamplerDesc);
 
    LLGLSamplerDescriptor lodSamplerDesc = g_defaultSamplerDesc;
  { lodSamplerDesc.mipMapLODBias = 3; }
  samplers[1] = llglCreateSampler(&lodSamplerDesc);

  LLGLSamplerDescriptor nearestSamplerDesc = g_defaultSamplerDesc;
  {
    nearestSamplerDesc.minFilter = LLGLSamplerFilterNearest;
    nearestSamplerDesc.magFilter = LLGLSamplerFilterNearest;
    nearestSamplerDesc.minLOD = 4;
    nearestSamplerDesc.maxLOD = 4;
  }
  samplers[2] = llglCreateSampler(&nearestSamplerDesc);

     // Create shaders
  const LLGLShaderDescriptor vertShaderDesc = {
      .type = LLGLShaderTypeVertex,
      .source = "./Example.vert",
      .sourceType = LLGLShaderSourceTypeCodeFile,
      .vertex.numInputAttribs = ARRAY_SIZE(vertexAttributes),
      .vertex.inputAttribs = vertexAttributes,
  };
  const LLGLShaderDescriptor fragShaderDesc = {
      .type = LLGLShaderTypeFragment,
      .source = "./Example.frag",
      .sourceType = LLGLShaderSourceTypeCodeFile};

     // Specify vertex attributes for vertex shader
  LLGLShader shaders[2] = {
      llglCreateShader(&vertShaderDesc),
      llglCreateShader(&fragShaderDesc),
  };

    for (int i = 0; i < 2; ++i) {
        LLGLReport shaderReport = llglGetShaderReport(shaders[i]);
        if (llglHasReportErrors(shaderReport)) {
        fprintf(stderr, "%s\n", llglGetReportText(shaderReport));
        return 1;
        }
    }


  // Create pipeline layout to describe the binding points
  const LLGLBindingDescriptor psoBindings[] = {
      {.name = "Scene",
       .type = LLGLResourceTypeBuffer,
       .bindFlags = LLGLBindConstantBuffer,
       .stageFlags = LLGLStageVertexStage,
       .slot.index = 1},
      {.name = "colorMap",
       .type = LLGLResourceTypeTexture,
       .bindFlags = LLGLBindSampled,
       .stageFlags = LLGLStageFragmentStage,
       .slot.index = 2},
      {.name = "samplerState",
       .type = LLGLResourceTypeSampler,
       .bindFlags = 0,
       .stageFlags = LLGLStageFragmentStage,
       .slot.index = 2},
  };
  const LLGLPipelineLayoutDescriptor psoLayoutDesc = {
      .numBindings = ARRAY_SIZE(psoBindings), .bindings = psoBindings};
  LLGLPipelineLayout pipelineLayout = llglCreatePipelineLayout(&psoLayoutDesc);

  // Create graphics pipeline
  const LLGLGraphicsPipelineDescriptor pipelineDesc = {
      .pipelineLayout = pipelineLayout,
      .vertexShader = shaders[0],
      .fragmentShader = shaders[1],
      .renderPass = llglGetRenderTargetRenderPass(
          LLGL_GET_AS(LLGLRenderTarget, g_swapChain)),
      .primitiveTopology = LLGLPrimitiveTopologyTriangleList,
      .depth.testEnabled = true,
      .depth.writeEnabled = true,
      .depth.compareOp = LLGLCompareOpLess,
      .rasterizer = {.multiSampleEnabled = true},
      .blend.targets[0].colorMask = LLGLColorMaskAll,
  };
  LLGLPipelineState pipeline = llglCreateGraphicsPipelineState(&pipelineDesc);

      // Link shader program and check for errors
  LLGLReport pipelineReport = llglGetPipelineStateReport(pipeline);
  if (llglHasReportErrors(pipelineReport)) {
    fprintf(stderr, "%s\n", llglGetReportText(pipelineReport));
    return 1;
  }
    // Scene state
  float rotation = -20.0f;

      // Enter main loop
  while (example_poll_events()) {
    // Update scene by mouse events
    if (key_pressed(LLGLKeyLButton))
      rotation += mouse_movement_x() * 0.5f;

    // Begin recording commands
    llglBegin(g_commandBuffer);
    {
      // Update scene constant buffer
      SceneConstants scene;
      {
        matrix_load_identity(scene.wMatrix);
        matrix_translate(scene.wMatrix, 0.0f, 0.0f, 5.0f);
        matrix_rotate(scene.wMatrix, 0.0f, 1.0f, 0.0f, DEG2RAD(rotation));

        matrix_mul(scene.wvpMatrix, g_projection, scene.wMatrix);
      }
      llglUpdateBuffer(sceneBuffer, 0, &scene, sizeof(scene));

      // Set vertex and index buffers
      llglSetVertexBuffer(vertexBuffer);
      llglSetIndexBuffer(indexBuffer);

      // Set the swap-chain as the initial render target
      llglBeginRenderPass(LLGL_GET_AS(LLGLRenderTarget, g_swapChain));
      {
        // Clear color and depth buffers
        llglClear(LLGLClearColorDepth, &g_defaultClear);
        llglSetViewport(&g_viewport);

        // Set graphics pipeline
        llglSetPipelineState(pipeline);

        llglSetResource(0, LLGL_GET_AS(LLGLResource, sceneBuffer));
        llglSetResource(1, LLGL_GET_AS(LLGLResource, colorTexture));
        llglSetResource(2, LLGL_GET_AS(LLGLResource, samplers[0]));

        // Draw cube mesh with index and vertex buffers
        llglDrawIndexed(indexCount, 0);
      }
      llglEndRenderPass();
    }
    llglEnd();

    // Present the result on the screen
    llglPresent(g_swapChain);
  }

  // Clean up
  releaseSystem();
    stbi_image_free(imageBuffer);
  return 0;







}