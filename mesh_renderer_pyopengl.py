import glfw
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
import numpy as np
import trimesh


def load_mesh(path):
    # Load model (可能是 Trimesh 或 Scene)
    mesh_or_scene = trimesh.load(path)

    # 如果是 Scene（多個子物件），合併成單一 mesh
    if isinstance(mesh_or_scene, trimesh.Scene):
        mesh = mesh_or_scene.dump().sum()
    else:
        mesh = mesh_or_scene
    
    # 轉為 numpy 格式
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)
    
    # Optional texture support
    if mesh.visual.kind == 'texture':
        uvs = np.array(mesh.visual.uv, dtype=np.float32)
        image = mesh.visual.material.image  # PIL Image
    else:
        uvs = None
        image = None

    return vertices, faces, uvs, image

def main():
    if not glfw.init():
        return
    window = glfw.create_window(800, 800, "Simple Mesh Viewer", None, None)
    glfw.make_context_current(window)

    glEnable(GL_DEPTH_TEST)

    # Set projection matrix ONCE
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(40, 1.0, 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)
    vertices, faces = load_mesh("/home/syjintw/Desktop/NEU/dataset/test_mesh/hotdog_Y_Z.obj")  # ← 替換成你的模型

    while not glfw.window_should_close(window):
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Reset MODELVIEW matrix every frame
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glTranslatef(0, 0, -3)
        glScalef(1.0, 1.0, 1.0)

        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_TRIANGLES)
        for face in faces:
            for idx in face:
                glVertex3fv(vertices[idx])
        glEnd()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
