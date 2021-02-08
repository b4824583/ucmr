import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from absl import app, flags
import pymesh

FLAGS = flags.FLAGS

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('fileType',"", 'file type ,only input obj or off')
flags.DEFINE_string('fileName',"", 'input file name ex:bird.obj , bird.off')
flags.DEFINE_string("textureFileName","","input texture name ex:bird_texture.off")

# flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
# flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')

def load_obj_file(path, device='cpu'):
    pmesh = pymesh.load_mesh(path)
    verts = pmesh.vertices
    faces = pmesh.faces
    elem = {
        'verts':verts,
        'faces':faces,
    }
    if pmesh.has_attribute('corner_texture'):
        faces_uv = pmesh.get_face_attribute('corner_texture')
        faces_uv = faces_uv.reshape(faces.shape + (2,))
        elem['faces_uv'] = faces_uv
    return elem


def getAreaRatio(element):
    verts=element["verts"]
    faces=element["faces"]
    faces_uv=element["faces_uv"]
    # if(fileType=="off"):
    #     verts_uv=element["verts_uv"]

    # area_ratio_list=np.empty([len(faces_uv)],dtype=float)
    area_3d_list=np.empty([len(faces_uv)],dtype=float)
    area_2d_list=np.empty([len(faces_uv)],dtype=float)
    for i in range(len(faces)):
        face=faces[i]
        area_3d_list[i]=triangle_area3D(verts, face)
        # if(fileType=="off"):
        #     uv_vert1=verts_uv[faces_uv[i][0]]
        #     uv_vert2=verts_uv[faces_uv[i][1]]
        #     uv_vert3=verts_uv[faces_uv[i][2]]
        # elif(fileType=="obj"):
        uv_vert1=faces_uv[i][0]
        uv_vert2=faces_uv[i][1]
        uv_vert3=faces_uv[i][2]
        area_2d_list[i]=triangle_area2D(uv_vert1,uv_vert2,uv_vert3)
        # area_ratio_list[i]=area_3d_list[]/area_2d_list[i]
    # print("area 2d sum:\t"+str(area_2d_list.sum()))
    # print("area 3d sum:\t"+str(area_3d_list.sum()))
    normalize=(area_2d_list.sum() / area_3d_list.sum()) ** 0.5
    # print((area_2d_list.sum() / area_3d_list.sum())**0.5)

    # normalize=area3D2D["area_ratio"]**0.5
    return normalize
def drawHistogram(meshL2):

    names=range(len(meshL2))
    values=meshL2
    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.bar(names, values)
    # plt.subplot(132)
    # plt.scatter(names, values)
    # plt.subplot(133)
    # plt.plot(names, values)
    plt.suptitle('Categorical Plotting')
    plt.show()

def drawGaussionDistribution(meshE2):
    standard_deviation=np.std(meshE2,ddof=1)
    print("standard deviation\t"+str(standard_deviation))
    mu = meshE2.mean()
    variance = standard_deviation**2
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()
    return 0
def getL2_total(meshL2,mesh_area):
    L2_total=0.0
    for i in range(len(meshL2)):
        L2_total += ((meshL2[i] ** 2) * mesh_area[i])
    L2_total = (L2_total / mesh_area.sum()) ** 0.5

    return L2_total
def getE2_total(conformal,area):
    E2_total=conformal.sum()+area.sum()
    return E2_total
def LInfinity(a,b,c):
    b_square_and_4times=(b**2)*4
    a_minus_c_and_square=(a-c)**2
    a_plus_c=a+c
    L_infinity=((a_plus_c+((a_minus_c_and_square+b_square_and_4times)**0.5))/2)**0.5
    return L_infinity
def triangle_area3D(verts,face):
    vert1=verts[face[0]]
    vert2=verts[face[1]]
    vert3=verts[face[2]]
    x,y,z=0,1,2
    edge1=((vert1[x]-vert2[x])**2+(vert1[y]-vert2[y])**2+(vert1[z]-vert2[z])**2)**0.5
    edge2=((vert1[x]-vert3[x])**2+(vert1[y]-vert3[y])**2+(vert1[z]-vert3[z])**2)**0.5
    vert1_x,vert1_y,vert1_z=vert1[x],vert1[y],vert1[z]
    vert2_x,vert2_y,vert2_z=vert2[x],vert2[y],vert2[z]
    vert3_x,vert3_y,vert3_z=vert3[x],vert3[y],vert3[z]
    vector_1x2=np.array([(vert1_x-vert2_x),(vert1_y-vert2_y),(vert1_z-vert2_z)])
    vector_1x3=np.array([(vert1_x-vert3_x),(vert1_y-vert3_y),(vert1_z-vert3_z)])
    unit_vector_1x2=vector_1x2/np.linalg.norm(vector_1x2)#this line error
    unit_vector_1x3=vector_1x3/np.linalg.norm(vector_1x3)

    dot_product=np.dot(unit_vector_1x2,unit_vector_1x3)
    angle=np.arccos(dot_product)
    sin=math.sin(angle)
    area=0.5*edge1*edge2*sin
    return area
def triangle_area2D(a,b,c):
    x1,y1=a[0],a[1]
    x2,y2=b[0],b[1]
    x3,y3=c[0],c[1]
    return abs(0.5*(((x2-x1)*(y3-y1))-((x3-x1)*(y2-y1))))
def L2(a,c):
    L2=((a+c)/2)**0.5
    return L2
def E2(a,b,c):
    E_conformal=((a-c)**2)+(4*(b**2))
    E_area=(a+c-2)**2
    # print("E conformal:\t"+str(E_conformal))
    # print("E area:\t\t"+str(E_area))
    return E_conformal,E_area
    # print("area rate:\t\t"+str(faceArea/faceAreaUV))

def metricTensorP(verts,face,face_uv_with_position,normalize):
    vertex1=verts[face[0]]
    vertex2=verts[face[1]]
    vertex3=verts[face[2]]
    uvVertex1=face_uv_with_position[0]
    uvVertex2=face_uv_with_position[1]
    uvVertex3=face_uv_with_position[2]
    faceAreaUV=triangle_area2D(uvVertex1,uvVertex2,uvVertex3)

    # faceAreaUV=normalize*faceAreaUV
    x,y=0,1
    q1,q2,q3=vertex1,vertex2,vertex3
    s1,s2,s3=uvVertex1[x],uvVertex2[x],uvVertex3[x]
    t1,t2,t3=uvVertex1[y],uvVertex2[y],uvVertex3[y]
    Ps=(q1*(t2-t3) + q2*(t3-t1)+q3*(t1-t2))/(2*faceAreaUV)#P Partial Derivative S
    Pt=(q1*(s3-s2) + q2*(s1-s3) + q3*(s2-s1))/(2*faceAreaUV)#P Partial Derivative T
    Ps=normalize*Ps
    Pt=normalize*Pt
    a=np.dot(Ps,Ps)
    b=np.dot(Ps,Pt)
    c=np.dot(Pt,Pt)
    return a,b,c
def computeL2_and_E2(verts,faces,faces_uv):
    #----------------show verts type
    # print("verts")
    # print(type(verts))
    # print("faces")
    # print(type(faces))
    # print("faces_uv")
    # print(type(faces_uv))
    #----------------------
    texture_face_num = len(faces_uv)
    meshL2 = np.empty([texture_face_num], dtype=float)
    mesh_area = np.empty([texture_face_num], dtype=float)
    meshLInfinity = np.empty([texture_face_num], dtype=float)
    element={"verts":verts,"faces":faces,"faces_uv":faces_uv}
    normalize = getAreaRatio(element)

    meshE_conformal = np.empty([texture_face_num], dtype=float)
    meshE_area = np.empty([texture_face_num], dtype=float)
    for i in range(len(faces)):
        a, b, c = metricTensorP(verts, faces[i], faces_uv[i], normalize)
        meshL2[i] = L2(a, c)
        meshE_conformal[i], meshE_area[i] = E2(a, b, c)
        meshLInfinity[i] = LInfinity(a, b, c)
        mesh_area[i] = triangle_area3D(element["verts"], faces[i])
    # L2_total = getL2_total(meshL2, mesh_area)
    # print("L2 total:\t" + str(L2_total))
    # print("L infinity:\t" + str(meshLInfinity.max()))
    # print("L2 max:\t\t" + str(meshL2.max()))
    # print("L2 min:\t\t" + str(meshL2.min()))

    L2_total = getL2_total(meshL2, mesh_area)
    return L2_total


    # print("L2 total:\t" + str(L2_total))
    # print("L infinity:\t" + str(meshLInfinity.max()))
    #
    # E2_total = meshE_conformal.sum() + meshE_area.sum()
    # print("E2 total:\t" + str(E2_total))
    # E2_average = E2_total / texture_face_num
    # print("E2 average:\t" + str(E2_average))

