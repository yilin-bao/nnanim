from OpenGL.GL import *
from OpenGL.GLUT import *

# 旋转角度
angle = 0

def draw_cuboid():
    # 绘制底面
    glBegin(GL_QUADS)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glEnd()

    # 绘制顶面
    glBegin(GL_QUADS)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glEnd()

    # 绘制前面
    glBegin(GL_QUADS)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glEnd()

    # 绘制后面
    glBegin(GL_QUADS)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glEnd()

    # 绘制左侧
    glBegin(GL_QUADS)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glEnd()

    # 绘制右侧
    glBegin(GL_QUADS)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glEnd()

def display():
    global angle

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # 透视投影
    fovy = 45.0
    aspect = 800 / 600
    zNear = 0.1
    zFar = 50.0
    gluPerspective(fovy, aspect, zNear, zFar)

    # 平移和旋转相机
    glTranslatef(0, 0, -5)
    glRotatef(angle, 0, 1, 0)

    # 绘制长方体
    draw_cuboid()

    glutSwapBuffers()
    angle += 0.5  # 每帧旋转角度

def reshape(width, height):
    glViewport(0, 0, width, height)

def main():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("OpenGL Cuboid in Python")
    
    glEnable(GL_DEPTH_TEST)

    glutDisplayFunc(display)
    glutReshapeFunc(reshape)

    glutMainLoop()

if __name__ == "__main__":
    main()