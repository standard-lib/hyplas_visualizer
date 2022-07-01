# -*- coding: utf-8 -*-
"""
nodal_point_coordinates.plt，element_connectivities.pltに従った格子を表示する．
要素をtoColorの大きさに従って色をつける．
またdraw_outlineがTrueの場合は，要素の輪郭に線を入れる，
<色付けについての補足>
4節点以上の要素の場合，全ての節点の座標x,yの平均xo, yoに新たな点（中心点）を取る．
中心点の物理量は全ての節点の物理量の平均とした上で，隣り合う2節点と中心点を結ぶ三角形を描く．
面の色は三角形の頂点の物理量をもとにtricontourf関数で補間して描く．
このため，形状関数で想定されるような分布にはもちろんなっていない．
そもそも要素間の平均を取った節点の物理量をもとに表されているので正確な応力分布の表示は無理．

Created on Thu Jun 23 17:47:52 2022

@author: Naoki Matsuda
"""
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize 
import copy
import sys

#形状関数
N_Q4 = lambda xi,eta:\
    0.25* np.matrix([(1-xi)*(1-eta),\
                      (1+xi)*(1-eta),\
                      (1+xi)*(1+eta),\
                      (1-xi)*(1+eta)]);

N_Q8 = lambda xi,eta:\
    0.25*np.matrix([(1-xi)*(1-eta)*(-1-xi-eta),\
                     2*(1-xi**2)*(1-eta),\
                     (1+xi)*(1-eta)*(-1+xi-eta),\
                     2*(1+xi)*(1-eta**2),\
                     (1+xi)*(1+eta)*(-1+xi+eta),\
                     2*(1-xi**2)*(1+eta),\
                     (1-xi)*(1+eta)*(-1-xi+eta),\
                     2*(1-xi)*(1-eta**2)]);

# 形状関数の正規化座標xi,etaに関する勾配
grad_norm_N_Q4 = lambda xi, eta:\
    0.25* np.matrix([[-(1-eta), -(1-xi)],\
                     [  1-eta , -(1+xi)],\
                     [  1+eta ,   1+xi ],\
                     [-(1+eta),   1-xi]]);

grad_norm_N_Q8  = lambda xi, eta:\
    0.25*np.matrix([\
                 [(1-eta)*(2*xi+eta), (1-xi)*(2*eta+xi)],\
                 [-4*(1-eta)*xi     ,  -2*(1-xi**2)],\
                 [(1-eta)*(2*xi-eta), (1+xi)*(2*eta-xi)],\
                 [2*(1-eta**2)      , -4*(1+xi)*eta],\
                 [(1+eta)*(2*xi+eta), (1+xi)*(2*eta+xi)],\
                 [-4*(1+eta)*xi     ,  2*(1-xi**2)],\
                 [(1+eta)*(2*xi-eta), (1-xi)*(2*eta-xi)],\
                 [-2*(1-eta**2)     , -4*(1-xi)*eta]]);

# Dマトリックス
D_plainstress = lambda E,nu:  E/(1-nu**2)*np.matrix([[1, nu, 0],[nu,1,0],[0,0,(1-nu)/2]]);
D_plainstrain = lambda E,nu:  E/((1+nu)*(1-2*nu))*np.matrix([[1-nu, nu, 0],[nu,1-nu,0],[0,0,(1-2*nu)/2]]);

def readSfile(filename, points):
    """
    各種応力（と角度）のｐｌｔファイルを読み込む
    異なるグループに同じ節点が入っている場合，グループ間で単純に平均する．節点につながっている要素の数は考慮されない．

    Parameters
    ----------
    filename : string
        pltファイルのファイル名（.pltも含めたフルファイル名）.
    points : int
        読み込むファイルに含まれる節点の数.複数のグループがある場合，節点の数はpltファイルの行数よりも少なくなる．

    Returns
    -------
    S : ListのList
        [ノード番号, 物理量]をリストにしたもの.

    """
    f = open(filename, 'r')
    print('Reading ', filename)
    data = f.read()
    f.close()
    data_in_str = [s.split() for s in data.splitlines()]
    S_all = [[int(line[0]), int(line[1]), float(line[2])] for line in data_in_str]
    # print(S_all)
    S = [];
    for icoord in range(points):
        vals = [S_all[s][2] for s, x in enumerate(S_all) if x[1] == icoord+1 ]
        S.append([icoord+1, sum(vals)/len(vals)]);
        # print(vals)
        # print(sum(vals)/len(vals))
    return S

def fileread():
    """
    .pltファイルを読み込んでリストの形で返す

    Returns
    -------
    original_nodal_point_coordinates : ListのList
        ノードの座標（変形前），[ノード番号,x座標，y座標]のリスト
    displacements : ListのList
        ノードの変位．[ノード番号,x変位，y変位]のリスト
    element_connectivities : ListのList
        要素の接続情報[要素番号,グループ番号,対応要素を構成するノード番号のリスト]のリスト
    S_xx : ListのList
        ノードごとに平均されたSxx.
    S_yy : ListのList
        ノードごとに平均されたSyy.
    S_xy : ListのList
        ノードごとに平均されたSxy.
    S_max : ListのList
        ノードごとに平均されたSmax.
    S_min : ListのList
        ノードごとに平均されたSmin.
    Angle : TYPE
        ノードごとに平均されたAngle.

    """
    # file reads
    f = open('Nodal_point_co-ordinates.plt', 'r')
    data = f.read()
    f.close()
    data_in_str = [s.split() for s in data.splitlines()]
    original_nodal_point_coordinates = [[float(s) for s in line] for line in data_in_str]

    f = open('Displacement.plt', 'r')
    data = f.read()
    f.close()
    data_in_str = [s.split() for s in data.splitlines()]
    displacements = [[float(s) for s in line] for line in data_in_str]

    f = open('Element_connectivities.plt', 'r')
    data = f.read()
    f.close()
    data_in_str = [s.split() for s in data.splitlines()]
    element_connectivities = [[int(s) for s in line] for line in data_in_str]
    
    S_xx = readSfile('S-xx.plt', len(original_nodal_point_coordinates))
    S_yy = readSfile('S-yy.plt', len(original_nodal_point_coordinates))
    S_xy = readSfile('S-xy.plt', len(original_nodal_point_coordinates))
    S_max = readSfile('S-max.plt', len(original_nodal_point_coordinates))
    S_min = readSfile('S-min.plt', len(original_nodal_point_coordinates))
    Angle = readSfile('Angle.plt', len(original_nodal_point_coordinates))

    return original_nodal_point_coordinates, displacements, element_connectivities, S_xx, S_yy, S_xy, S_max, S_min, Angle

def deform_nodalpoint(original_nodal_point_coordinates, displacements, magnification):
    """
    変位をmagnification倍にして，節点座標を計算する

    Parameters
    ----------
    original_nodal_point_coordinates : ListのList
        ファイル読み込みしたノード座標
    displacements : ListのList
        ファイル読み込みしたノードごとの変位
    magnification : float
        変位を何倍してオリジナルノード座標に足すか

    Returns
    -------
    nodal_point_coordinates : ListのList
        ノードごとの変形後の座標

    """
    nodal_point_coordinates = copy.deepcopy(original_nodal_point_coordinates);
    for inode in range(len(nodal_point_coordinates)):
        nodal_point_coordinates[inode][1] += magnification*displacements[inode][1];
        nodal_point_coordinates[inode][2] += magnification*displacements[inode][2];
    # print(nodal_point_coordinates)
    return nodal_point_coordinates


def deformplot(fig, ax, nodal_point_coordinates, element_connectivities, toColor, draw_outline=True, title = "", unit = "[MPa]"):
    """
    nodal_point_coordinates，element_connectivitiesに従った格子を表示する．
    要素をtoColorの大きさに従って色をつける．
    またdraw_outlineがTrueの場合は，要素の輪郭に線を入れる，
    <色付けについての補足>
    4節点以上の要素の場合，全ての節点の座標x,yの平均xo, yoに新たな点（中心点）を取る．
    中心点の物理量は全ての節点の物理量の平均とした上で，隣り合う2節点と中心点を結ぶ三角形を描く．
   　面の色は三角形の頂点の物理量をもとにtricontourf関数で補間して描く．
    このため，形状関数で想定されるような分布にはもちろんなっていない．
    そもそも要素間の平均を取った節点の物理量をもとに表されているので正確な応力分布の表示は無理．

    Parameters
    ----------
    fig : matplotlib.figure
        Figure.
    ax : matplotlib.axes.Axes
        Axis.
    nodal_point_coordinates : List of List
        ノードの座標値．変形後でも変形前でもどちらでも対応できる．
    element_connectivities : List of List
        要素の接続情報.
    toColor : List of List
        色付けをするための物理量．節点ごとの情報で与える.
    draw_outline : bool, optional
        要素の輪郭を描くかどうか．密な切り方をした場合だと，これをTrueにしていると見えない可能性がある．
        このような場合はこのオプションをFalseにする. The default is True.
    title : string, optional
        グラフのタイトル. The default is "".

    Returns
    -------
    clb : colorbar
        カラーバー.


    """
    
    outerframe_lines = []; #要素の輪郭を構成する線の情報を入れる．例えば4節点なら，[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]のリスト
    # 色付けするためのデータたち．あとでtricontourf(x, y, triangles, v... という形で使用する
    x = [l[1] for l in nodal_point_coordinates] # ノードごとのx座標.ここに後で四角形要素の中心点も追加する
    y = [l[2] for l in nodal_point_coordinates] # ノードごとのy座標.ここに後で四角形要素の中心点も追加する
    v = [l[1] for l in toColor] #ノードごとの物理量．.ここに後で四角形要素の中心点の情報も追加する
    triangles = []; # 三角形を構成するノードの番号を入れる．このノード番号は，x,y,vのインデックス（すなわちFEMの節点番号-1
    for element in element_connectivities:
        nx = [nodal_point_coordinates[s-1][1] for s in element[2:]] #elementを構成する節点のx座標のリスト
        ny = [nodal_point_coordinates[s-1][2] for s in element[2:]] #elementを構成する節点のy座標のリスト
        nv = [toColor[s-1][1] for s in element[2:]] #elementを構成する節点の物理量のリスト
        
        # 要素の輪郭は基本的にはnx, nyを並べたもの．一周してくるように，一番最初の座標を最後に付け加える．
        outerframe_lines.append(list(zip(nx+[nx[0]],ny+[ny[0]]))); 
        
        #三角形要素の場合，コンター図はそのまま出力．
        if( len(element[2:]) == 3 ):
            triangles.append([x-1 for x in element[2:]]); #要素を構成するノード番号をそのまま使って三角形にする．
        #四角形要素の場合，平均値の点を中心に辺ごとに三角形を構成して，その三角形に対してコンターを描く．
        else:
            #中心座標とその物理量の計算
            avgx = sum(nx)/len(nx);
            avgy = sum(ny)/len(ny);
            avgv = sum(nv)/len(nv);

            idxappend = len(x); # 新しい中心点の「ノード」番号に相当する．
            # 中心点を新しいノードとして追加．
            x.append(avgx);
            y.append(avgy);
            v.append(avgv);
            
            #中心点を取り囲むような三角形のノード番号リストを作る．
            vert1 = [x-1 for x in element[2:]];
            vert2 = vert1[1:]+[vert1[0]];
            vert3 = [idxappend]*len(vert1);
            triangles = triangles + [list(x) for x in zip(vert1,vert2,vert3)];
    # print('triangles')
    # print(triangles)
    
    # 色付けの最小値と最大値を与えられた物理量の最小値と最大値に設定する
    colmin = min(l[1] for l in toColor)
    colmax = max(l[1] for l in toColor)
    colorv = np.linspace(colmin, colmax, 100, endpoint=True) #カラーバーの色の刻み．100を小さくすると荒くなる
    
    # 色付きの面の描画
    tcf = ax.tricontourf(x, y, triangles, v, colorv, cmap='jet', norm=Normalize(vmin=colmin, vmax=colmax))
    clb = fig.colorbar(tcf, ax=ax) # カラーバーの表示
    
    # 要素の輪郭線の描画
    if(draw_outline):
        line_segments = LineCollection(outerframe_lines, colors="black", linewidths = 0.3);
        ax.add_collection(line_segments);
        
    # 軸設定
    plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
    plt.rcParams["mathtext.fontset"] = "stix" 
    plt.rcParams["font.size"] = 10
    ax.set_aspect('equal');
    ax.set_title(title);
    ax.set_xlabel('$\it{x}$ [mm]')
    ax.set_ylabel('$\it{y}$ [mm]')
    ax.autoscale_view()
    clb.set_label(unit)
    return

def make_triangles(original_nodal_point_coordinates, displacements, element_connectivities, nodal_point_coordinates, E, nu, analysis_type, xis):
    trixlist = [];
    triylist = [];
    trinodelist = []
    S_xxlist = [];
    S_yylist = [];
    S_xylist = [];
    for element in element_connectivities:
        element_nodeno = [s-1 for s in element[2:]]
        node_no = len(trixlist);
        original_node_coords = np.matrix([[original_nodal_point_coordinates[idx][1], original_nodal_point_coordinates[idx][2]] for idx in element_nodeno]).transpose();
        element_node_displacements = np.matrix([[displacements[idx][1],displacements[idx][2]] for idx in element_nodeno]);
        
        if len(element_nodeno) == 3: # 3節点三角形要素
            xi_eta = [[0, 0]];
            stresslist = calculate_stress(original_node_coords, element_node_displacements, element_nodeno, E, nu, analysis_type, xi_eta);
            S_xxlist += [stresslist[0,0]] * 3;
            S_yylist += [stresslist[0,1]] * 3;
            S_xylist += [stresslist[0,2]] * 3;
            
            trixlist += [nodal_point_coordinates[idx][1] for idx in element_nodeno];
            triylist += [nodal_point_coordinates[idx][2] for idx in element_nodeno];
            trinodelist += [[node_no, node_no+1, node_no+2]];

        elif len(element_nodeno) == 4 or len(element_nodeno) == 8: # 4節点/8節点四辺形要素
            xi_eta = [[xi, eta] for xi in xis for eta in xis];
            stresslist = calculate_stress(original_node_coords, element_node_displacements, element_nodeno, E, nu, analysis_type, xi_eta);
            S_xxlist += stresslist[:,0].tolist();
            S_yylist += stresslist[:,1].tolist();
            S_xylist += stresslist[:,2].tolist();
            
            deformed_node_coords = np.matrix([[nodal_point_coordinates[idx][1], nodal_point_coordinates[idx][2]] for idx in element_nodeno])
            deformed_coords = calculate_mesh(deformed_node_coords, element_nodeno, xi_eta);
            trixlist += [s[0] for s in deformed_coords[:,0].tolist()];
            triylist += [s[0] for s in deformed_coords[:,1].tolist()];
            
            trinodelist += make_element_triangles(node_no, xis);
            
        else:
            raise ValueError("ノード数の不適切なelementがあります");
    
    return trixlist, triylist, trinodelist, S_xxlist, S_yylist, S_xylist

def calculate_stress(original_node_coords, element_node_displacement, element_nodeno, E, nu, analysis_type, xi_eta):
    D = D_plainstress(E,nu) if analysis_type == "plain_stress" else D_plainstrain(E,nu);
    stresslist = np.empty(0);
    for xi,eta in xi_eta:
        if len(element_nodeno) == 3:
            A = np.vstack([np.matrix([1,1,1]), original_node_coords]).transpose();
            Ainv = np.linalg.inv(A);
            grad_whole_N = Ainv[1:3,0:3].transpose();
        else:
            grad_norm_N = grad_norm_N_Q4(xi, eta) if len(element_nodeno)==4 else grad_norm_N_Q8(xi, eta);
            Jinv = np.linalg.inv(np.dot( original_node_coords, grad_norm_N ));
            grad_whole_N = np.dot(grad_norm_N, Jinv);
        strains = np.zeros((3,1));
        # ここのforは汚い・・・
        for n in range(len(element_nodeno)):
            Bn = np.matrix([[grad_whole_N[n,0], 0], [0, grad_whole_N[n,1]], [grad_whole_N[n,1], grad_whole_N[n,0]]]);
            U = np.matrix([[displacements[element_nodeno[n]][1]],[displacements[element_nodeno[n]][2]]]);
            strains += np.dot(Bn, U);
        stresslist = np.append( stresslist, np.dot(D,strains));
    return stresslist.reshape([-1,3]);
 
def calculate_mesh(deformed_node_coords,  element_nodeno, xi_eta):
    N = [N_Q4(xi,eta) if len(element_nodeno)==4 else N_Q8(xi, eta) for xi,eta in xi_eta]
    return np.dot(N, deformed_node_coords);
        
def make_element_triangles(node_no, xis):
    divs = len(xis);
    triangles = [[node_no+i*divs+j, node_no+(i+1)*divs+j, node_no+i*divs+j+1 ] for i in range(divs-1) for j in range(divs-1)]
    triangles += [[node_no+i*divs+j+1, node_no+(i+1)*divs+j, node_no+(i+1)*divs+j+1 ] for i in range(divs-1) for j in range(divs-1)]
    return triangles

def make_outerframe(nodal_point_coordinates, element_connectivities, xis):
    outerframe_lines = [];
    for element in element_connectivities:
        element_nodeno = [s-1 for s in element[2:]]
        if len(element_nodeno) == 3 or len(element_nodeno) == 4:
            nx = [nodal_point_coordinates[s-1][1] for s in element[2:]] #elementを構成する節点のx座標のリスト
            ny = [nodal_point_coordinates[s-1][2] for s in element[2:]] #elementを構成する節点のy座標のリスト
            # 要素の輪郭は基本的にはnx, nyを並べたもの．一周してくるように，一番最初の座標を最後に付け加える．
            outerframe_lines.append(list(zip(nx+[nx[0]],ny+[ny[0]])));
        else:
            xis_wo_first = xis[1:];
            xi_eta = [[xi, -1] for xi in xis] + \
                     [[1, eta] for eta in xis_wo_first] + \
                     [[-xi, 1] for xi in xis_wo_first] + \
                     [[-1,-eta] for eta in xis_wo_first];
            deformed_node_coords = np.matrix([[nodal_point_coordinates[idx][1], nodal_point_coordinates[idx][2]] for idx in element_nodeno])
            lines_matrix = calculate_mesh( deformed_node_coords, element_nodeno, xi_eta);
            lines = [(s[0,0],s[0,1]) for s in lines_matrix];
            outerframe_lines.append(lines);
    return outerframe_lines

def deformplotbytruestress(fig, ax, trixlist, triylist, trinodelist, trivlist, outerframe_lines, title, unit = "[MPa]"):
    colmin = min(trivlist); colmin *= 0.999 if colmin > 0 else 1.001;
    colmax = max(trivlist); colmax *= 1.001 if colmax > 0 else 0.999;
    colorv = np.linspace(colmin, colmax, 100, endpoint=True) #カラーバーの色の刻み．100を小さくすると荒くなる
    tcf = ax.tricontourf(trixlist, triylist, trinodelist, trivlist, colorv, cmap='jet')
    # tcf = ax.tricontourf(trixlist, triylist, trinodelist, trivlist, colorv, cmap='jet', norm=Normalize(vmin=colmin, vmax=colmax))
    clb = fig.colorbar(tcf, ax=ax) # カラーバーの表示

    line_segments = LineCollection(outerframe_lines, colors="black", linewidths = 0.3);
    ax.add_collection(line_segments);
    
    # 軸設定
    plt.rcParams['font.family'] = 'Times New Roman' #全体のフォントを設定
    plt.rcParams["mathtext.fontset"] = "stix" 
    plt.rcParams["font.size"] = 10
    ax.set_aspect('equal');
    ax.set_title(title);
    ax.set_xlabel('$\it{x}$ [mm]')
    ax.set_ylabel('$\it{y}$ [mm]')
    ax.autoscale_view()
    clb.set_label(unit)
    return
    

# ここからmainプログラム


# 変位拡大率入力部
while True:
    try:
        s = input("変位の拡大率を入力：")
        magnification = float(s)
        break
    except EOFError as e:
        print(e)
        break
    except ValueError as e:
        print(e)

# 平面応力・平面ひずみ入力部
while True:
    try:
        s = input("解析条件を入力（平面応力：1，平面ひずみ:2）：")
        analysis_type_code = int(s)
        if(analysis_type_code > 2 or analysis_type_code < 1):
            continue
        break
    except EOFError as e:
        print(e)
        break
    except ValueError as e:
        print(e)
analysis_type = "plain_stress" if analysis_type_code == 1 else "plain_strain"

#ファイル読み込み    
original_nodal_point_coordinates, displacements, element_connectivities, S_xx, S_yy, S_xy, S_max, S_min, Angle = fileread();
# 変位をmagnification倍に拡大
nodal_point_coordinates = deform_nodalpoint(original_nodal_point_coordinates, displacements, magnification)
    
# 各物理量で色付けした出力
fig, ax = plt.subplots(dpi=200);
deformplot(fig, ax, nodal_point_coordinates, element_connectivities, S_xx, title='Normal stress in x-direction (averaged on nodes)')
# 要素間の線を消したい場合は下記を上記の代わりに実行
# deformplot(fig, ax, nodal_point_coordinates, element_connectivities, S_xx, title='Normal stress in x-direction', draw_outline=False)

fig, ax = plt.subplots(dpi=200);
deformplot(fig, ax, nodal_point_coordinates, element_connectivities, S_yy, title='Normal stress in y-direction (averaged on nodes)')

fig, ax = plt.subplots(dpi=200);
deformplot(fig, ax, nodal_point_coordinates, element_connectivities, S_xy, title='Shear stress in xy-direction (averaged on nodes)')

# fig, ax = plt.subplots();
# deformplot(fig, ax, nodal_point_coordinates, element_connectivities, S_max, title='maximum pricipal stress (averaged on nodes)')

# fig, ax = plt.subplots();
# deformplot(fig, ax, nodal_point_coordinates, element_connectivities, S_min, title='minimum pricipal stress (averaged on nodes)', unit='[MPa]')

# fig, ax = plt.subplots();
# deformplot(fig, ax, nodal_point_coordinates, element_connectivities, Angle, title='angle (averaged on nodes)', unit='[Degree]')


###### 節点で応力を平均化せず，節点変位と形状関数から応力分布を計算する場合の可視化（要素間で応力は不連続になる）
E = 210000.0;
nu = 0.3;

# xis = [-1.0, -0.5, 0.0, 0.5, 1.0];
xis = [-1.0, -np.sqrt(1/3), 0.0, np.sqrt(1/3), 1.0];
# xis = [-1.0, 1.0];

trixlist, triylist, trinodelist, S_xxlist, S_yylist, S_xylist = make_triangles(original_nodal_point_coordinates, displacements, element_connectivities, nodal_point_coordinates, E, nu, analysis_type, xis);
outerframe_lines = make_outerframe(nodal_point_coordinates, element_connectivities, xis);

fig, ax = plt.subplots(dpi=200);
deformplotbytruestress(fig, ax, trixlist, triylist, trinodelist, S_xxlist, outerframe_lines, title='Normal stress in x-direction')

fig, ax = plt.subplots(dpi=200);
deformplotbytruestress(fig, ax, trixlist, triylist, trinodelist, S_yylist, outerframe_lines, title='Normal stress in y-direction')

fig, ax = plt.subplots(dpi=200);
deformplotbytruestress(fig, ax, trixlist, triylist, trinodelist, S_xylist, outerframe_lines, title='Shear stress in xy-direction')

plt.show()
