import bpy
from mathutils import Vector

# 刪除場上的所有物體
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()
# 刪除場上的所有光源
bpy.ops.object.select_by_type(type='LIGHT')
bpy.ops.object.delete()
# 刪除場上的所有攝影機
bpy.ops.object.select_by_type(type='CAMERA')
bpy.ops.object.delete()

file_name = "result_ryota"

# 載入OBJ模型
bpy.ops.wm.obj_import(
filepath="..\\..\\PIFu-master\\results\\pifu_demo\\" + file_name + ".obj", 
directory="..\\..\\PIFu-master\\results\\pifu_demo\\", 
files=[{"name":file_name + ".obj", "name":file_name + ".obj"}])
# 獲取載入的物體
obj = bpy.context.selected_objects[0]
# 將物體位置置於原點
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

#讓模型底部再z=0位置
bpy.data.objects[file_name].location = (0,0,0)
bpy.data.objects[file_name].location.z = bpy.data.objects[file_name].location.z + (bpy.data.objects[file_name].dimensions.y / 2)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

model = bpy.data.objects[file_name]

# 进入编辑模式
bpy.context.view_layer.objects.active = model
bpy.ops.object.mode_set(mode='EDIT')
# 选择所有的顶点
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.remove_doubles()
bpy.ops.mesh.tris_convert_to_quads()
# 返回对象到对象模式
bpy.ops.object.mode_set(mode='OBJECT')

# 創建一個新的材質
mat = bpy.data.materials.new(name="MyMaterial")

# 清除材質的所有節點
mat.use_nodes = True
nodes = mat.node_tree.nodes
for node in nodes:
    nodes.remove(node)

# 創建一個Shader節點
shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
shader_node.location = (0, 0)

# 將Shader節點連接到輸出節點
output_node = nodes.new(type='ShaderNodeOutputMaterial')
output_node.location = (400, 0)
mat.node_tree.links.new(shader_node.outputs["BSDF"], output_node.inputs["Surface"])

# 將Shader節點連接到輸入節點
input_node = nodes.new(type="ShaderNodeVertexColor")
input_node.location = (-200, 0)
mat.node_tree.links.new(input_node.outputs["Color"], shader_node.inputs["Base Color"])

#新增一張要被烘焙的圖
bake_img = nodes.new(type="ShaderNodeTexImage")
bake_img.location = (-400, -200)
# 创建一个新的空图像
image_width = 2048  # 设置图像宽度
image_height = 2048  # 设置图像高度
new_image = bpy.data.images.new(name="bake_img", width=image_width, height=image_height, alpha=False)
# 将新图像分配给Image节点
bake_img.image = new_image

# 將材質分配給物體
if model.data.materials:
    model.data.materials[0] = mat
else:
    model.data.materials.append(mat)
    
#在著色模式下看的到材質
for area in bpy.context.screen.areas: 
    if area.type == 'VIEW_3D':
        for space in area.spaces: 
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
                
# 进入编辑模式
bpy.context.view_layer.objects.active = model
bpy.ops.object.mode_set(mode='EDIT')
# 选择所有的顶点
bpy.ops.mesh.select_all(action='SELECT')
# 执行智能UV拆解
bpy.ops.uv.smart_project(island_margin=0.01)
# 返回对象到对象模式
bpy.ops.object.mode_set(mode='OBJECT')

# 设置渲染引擎为Cycles
bpy.context.scene.render.engine = 'CYCLES'
# 进入烘焙模式
bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.margin = 5
bpy.ops.object.bake(type='DIFFUSE')
bpy.ops.object.bake_image()
mat.node_tree.links.new(bake_img.outputs["Color"], shader_node.inputs["Base Color"])
image_to_save = bpy.data.images['bake_img']
# 保存图像
image_to_save.save_render("../ModelOutput/bake_img.png")
    
key_point = ["頭頂","嘴唇","鎖骨","左肩","右肩","左肘","右肘","左腕","右腕","左手指尖",\
"右手指尖","骨盆","左大腿跟","右大腿跟","左膝","右膝","左腳踝","右腳踝","左腳底","右腳底",\
"左腳趾尖","右腳趾尖", "左腳跟", "右腳跟"]
key_point_location = {}

# 創建球的列表
spheres = []

openORclose = False  # 鏡像關

# 創建 24 顆 UV 球，並將它們加入列表
for i in range(24):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=(0, 0, 0))
    sphere = bpy.context.active_object
    sphere.name = key_point[i]
    spheres.append(sphere)
    
def calculate_middle_points(p1, p2):
    dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    unit_dx, unit_dy, unit_dz = dx / 4, dy / 4, dz / 4
    print(unit_dx, unit_dy, unit_dz)
    middle_points = [
        (p1[0] + unit_dx, p1[1] + unit_dy, p1[2] + unit_dz),
        (p1[0] + 2 * unit_dx, p1[1] + 2 * unit_dy, p1[2] + 2 * unit_dz),
        (p1[0] + 3 * unit_dx, p1[1] + 3 * unit_dy, p1[2] + 3 * unit_dz)
    ]
    
    return middle_points

def mirror_movement(scene):
    if openORclose:
        bpy.data.objects.get("左肩").location = bpy.data.objects.get("右肩").location
        bpy.data.objects.get("左肩").location.x = -bpy.data.objects.get("右肩").location.x
        bpy.data.objects.get("左肘").location = bpy.data.objects.get("右肘").location
        bpy.data.objects.get("左肘").location.x = -bpy.data.objects.get("右肘").location.x
        bpy.data.objects.get("左腕").location = bpy.data.objects.get("右腕").location
        bpy.data.objects.get("左腕").location.x = -bpy.data.objects.get("右腕").location.x
        bpy.data.objects.get("左手指尖").location = bpy.data.objects.get("右手指尖").location
        bpy.data.objects.get("左手指尖").location.x = -bpy.data.objects.get("右手指尖").location.x
        bpy.data.objects.get("左大腿跟").location = bpy.data.objects.get("右大腿跟").location
        bpy.data.objects.get("左大腿跟").location.x = -bpy.data.objects.get("右大腿跟").location.x
        bpy.data.objects.get("左膝").location = bpy.data.objects.get("右膝").location
        bpy.data.objects.get("左膝").location.x = -bpy.data.objects.get("右膝").location.x
        bpy.data.objects.get("左腳踝").location = bpy.data.objects.get("右腳踝").location
        bpy.data.objects.get("左腳踝").location.x = -bpy.data.objects.get("右腳踝").location.x
        bpy.data.objects.get("左腳底").location = bpy.data.objects.get("右腳底").location
        bpy.data.objects.get("左腳底").location.x = -bpy.data.objects.get("右腳底").location.x
        bpy.data.objects.get("左腳趾尖").location = bpy.data.objects.get("右腳趾尖").location
        bpy.data.objects.get("左腳趾尖").location.x = -bpy.data.objects.get("右腳趾尖").location.x
        bpy.data.objects.get("左腳跟").location = bpy.data.objects.get("右腳跟").location
        bpy.data.objects.get("左腳跟").location.x = -bpy.data.objects.get("右腳跟").location.x
    
class SimplePanel(bpy.types.Panel):
    bl_label = "建模操作面板"
    bl_idname = "PT_SimplePanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'GetModel'

    def draw(self, context):
        layout = self.layout

        # 添加第一個按鈕
        if context.scene.enable_button1:
            layout.operator("wm.my_operator1")

        # 添加第二個按鈕
        if context.scene.enable_button2:
            layout.operator("wm.my_operator2")

        # 添加第三個按鈕
        if context.scene.enable_button3:
            layout.operator("wm.my_operator3")
            
        # 添加第四個按鈕
        if context.scene.enable_button4:
            layout.operator("wm.my_operator4")
        
class SimpleOperator1(bpy.types.Operator):
    bl_idname = "wm.my_operator1"
    bl_label = "紀錄關節位置"

    def execute(self, context):
        bpy.types.Scene.enable_button1 = bpy.props.BoolProperty(name="禁用按鈕 1", default=False)
        bpy.types.Scene.enable_button2 = bpy.props.BoolProperty(name="禁用按鈕 2", default=False)
        bpy.types.Scene.enable_button3 = bpy.props.BoolProperty(name="啟用按鈕 3", default=True)
        
        for i in range(24):
            key_point_location[key_point[i]] = bpy.data.objects.get(key_point[i]).location.copy() #cpoy要加以免錯亂
        # 刪除 spheres 列表中的所有球體物件
        for sphere in spheres:
            bpy.data.objects.remove(sphere, do_unlink=True)
        for i in range(24):
            print(key_point[i], key_point_location[key_point[i]])
        #key_point_location["頭頂"]=(0.0087, 0.0206, 1.8094)
        #key_point_location["嘴唇"]=(0.0000, 0.0206, 1.6064)
        #key_point_location["鎖骨"]=(-0.0043, 0.0314, 1.4183)
        #key_point_location["左肩"]=(0.1895, 0.0350, 1.3223)
        #key_point_location["右肩"]=(-0.2286, 0.0350, 1.3403)
        #key_point_location["左肘"]=(0.4029, 0.0222, 1.1911)
        #key_point_location["右肘"]=(-0.4081, 0.0119, 1.2039)
        #key_point_location["左腕"]=(0.6015, 0.0285, 1.0563)
        #key_point_location["右腕"]=(-0.6113, -0.0139, 1.0397)
        #key_point_location["左手指尖"]=(0.7182, -0.0639, 0.9591)
        #key_point_location["右手指尖"]=(-0.7264, -0.0930, 0.9380)
        #key_point_location["骨盆"]= (-0.0269, 0.0551, 0.8926)
        #key_point_location["左大腿跟"]=(0.0868, 0.0169, 0.7751)
        #key_point_location["右大腿跟"]=(-0.1479, 0.0169, 0.7751)
        #key_point_location["左膝"]=(0.1257, 0.0134, 0.4903)
        #key_point_location["右膝"]= (-0.2024, 0.0004, 0.4903)
        #key_point_location["左腳踝"]=(0.1813, 0.0844, 0.1589)
        #key_point_location["右腳踝"]=(-0.2825, 0.0953, 0.1705)
        #key_point_location["左腳底"]=(0.2384, 0.0288, 0.0476)
        #key_point_location["右腳底"]=(-0.3389, 0.0454, 0.0749)
        #key_point_location["左腳趾尖"]= (0.2396, -0.0147, 0.0131)
        #key_point_location["右腳趾尖"]=(-0.3475, 0.0233, 0.0335)
        #key_point_location["左腳跟"]=(0.1798, 0.1395, 0.1326)
        #key_point_location["右腳跟"]= (-0.2701, 0.1573, 0.1565)
        fbx_file_path = "../ezezezbone2.fbx"
        bpy.ops.import_scene.fbx(filepath=fbx_file_path)
        return {'FINISHED'}

class SimpleOperator2(bpy.types.Operator):
    bl_idname = "wm.my_operator2"
    bl_label = "開/關鏡像"

    def execute(self, context):
        global openORclose  # 宣告 openORclose 為全局變數
        openORclose = not openORclose  # 轉換開或關
        print(openORclose)
        if openORclose:
            #self.bl_label = "關閉鏡像"
            bpy.app.handlers.depsgraph_update_post.append(mirror_movement)
        else:
            #self.bl_label = "開啟鏡像"
            bpy.app.handlers.depsgraph_update_post.remove(mirror_movement)  # 移除事件處理
        return {'FINISHED'}

class SimpleOperator3(bpy.types.Operator):
    bl_idname = "wm.my_operator3"
    bl_label = "骨架對應"

    def execute(self, context):
        bpy.types.Scene.enable_button3 = bpy.props.BoolProperty(name="禁用按鈕 3", default=False)
        bpy.types.Scene.enable_button4 = bpy.props.BoolProperty(name="啟用按鈕 4", default=True)
        
        armature = bpy.context.active_object # 取得目前的活動物體（一般情況下是骨架物體）
        if armature.type == "ARMATURE": # 確保目前的活動物體是骨架
            armature_bones = armature.pose.bones # 取得骨架物體的骨骼
            bpy.ops.object.mode_set(mode='EDIT') # 進入骨頭的編輯模式                
            armature = bpy.data.objects["metarig"]
            matrix_world_inv = armature.matrix_world.inverted()
            matrix_world = armature.matrix_world
            #for i in range(22):
                #print(key_point[i], matrix_world_inv @ key_point_location[key_point[i]])
            # 選取指定的骨頭
            bone = armature.data.edit_bones["spine"] #中1 頭
            # 調整骨頭的位置
            head_pos = Vector(key_point_location["骨盆"]) #球-根
            bone.head = matrix_world_inv @ head_pos 
            
            bone = armature.data.edit_bones["spine.003"] #中4 尾
            spine2rigshoulder = (matrix_world @ bone.tail) - (matrix_world @ armature.data.edit_bones["shoulder.R"].head)
            spine2leftshoulder = (matrix_world @ bone.tail) - (matrix_world @ armature.data.edit_bones["shoulder.L"].head)
            print(spine2rigshoulder, spine2leftshoulder)
            tail_pos = Vector(key_point_location["鎖骨"]) #球-喉
            bone.tail = matrix_world_inv @ tail_pos
            
            middle_points = calculate_middle_points(head_pos, tail_pos) #三點中點

            bone = armature.data.edit_bones["spine"] #中 1
            bone.tail = matrix_world_inv @ Vector(middle_points[0])

            bone = armature.data.edit_bones["spine.001"]#中 2
            bone.head = matrix_world_inv @ Vector(middle_points[0])
            bone.tail = matrix_world_inv @ Vector(middle_points[1])

            bone = armature.data.edit_bones["spine.002"]#中 3
            bone.head = matrix_world_inv @ Vector(middle_points[1])
            bone.tail = matrix_world_inv @ Vector(middle_points[2])

            bone = armature.data.edit_bones["spine.004"]#中 5
            neck = matrix_world @ bone.tail
            bone.head = matrix_world_inv @ tail_pos
            bone.tail = matrix_world_inv @ Vector(key_point_location["嘴唇"]) # 球-臉
            newneck = matrix_world @ bone.tail

            bone = armature.data.edit_bones["spine.005"]#頭 
            bone.head = matrix_world_inv @ Vector(key_point_location["嘴唇"])
            bone.tail = matrix_world_inv @ Vector(key_point_location["頭頂"]) # 球-頭

            bone = armature.data.edit_bones["shoulder.L"]#左肩
            bone.head = matrix_world_inv @ (Vector(key_point_location["鎖骨"]) - spine2leftshoulder)
            bone.tail = matrix_world_inv @ Vector(key_point_location["左肩"])
            
            bone = armature.data.edit_bones["upper_arm.L"]#左肘
            bone.head = matrix_world_inv @ Vector(key_point_location["左肩"])
            bone.tail = matrix_world_inv @ Vector(key_point_location["左肘"])
            
            bone = armature.data.edit_bones["forearm.L"]#左腕
            bone.tail = matrix_world_inv @ Vector(key_point_location["左腕"])
            
            bone = armature.data.edit_bones["hand.L"]#左手指尖
            bone.tail = matrix_world_inv @ Vector(key_point_location["左手指尖"])

            bone = armature.data.edit_bones["shoulder.R"]#右肩
            bone.head = matrix_world_inv @ (Vector(key_point_location["鎖骨"]) - spine2rigshoulder)
            bone.tail = matrix_world_inv @ Vector(key_point_location["右肩"])
            
            bone = armature.data.edit_bones["upper_arm.R"]#右肘
            bone.head = matrix_world_inv @ Vector(key_point_location["右肩"])
            bone.tail = matrix_world_inv @ Vector(key_point_location["右肘"])
            
            bone = armature.data.edit_bones["forearm.R"]#右腕
            bone.tail = matrix_world_inv @ Vector(key_point_location["右腕"])
            
            bone = armature.data.edit_bones["hand.R"]#右手指尖
            bone.tail = matrix_world_inv @ Vector(key_point_location["右手指尖"])
            
            bone = armature.data.edit_bones["pelvis.L"]
            bone.head = matrix_world_inv @ Vector(key_point_location["骨盆"])
            bone.tail = matrix_world_inv @ Vector(key_point_location["左大腿跟"])
            bone = armature.data.edit_bones["thigh.L"]#左腿1
            bone.head = matrix_world_inv @ (Vector(key_point_location["左大腿跟"])) #球 左腳1
            bone.tail = matrix_world_inv @ (Vector(key_point_location["左膝"])) #球 左腳2
            bone = armature.data.edit_bones["shin.L"]#左腿2
            bone.tail = matrix_world_inv @ (Vector(key_point_location["左腳踝"])) #球 左腳3
            bone = armature.data.edit_bones["foot.L"]#左腳
            bone.tail = matrix_world_inv @ (Vector(key_point_location["左腳底"])) #球 左腳4
            bone = armature.data.edit_bones["toe.L"]#左腳趾
            bone.head = matrix_world_inv @ Vector(key_point_location["左腳底"])
            bone.tail = matrix_world_inv @ (Vector(key_point_location["左腳趾尖"]))#球 左腳4
            bone = armature.data.edit_bones["heel.02.L"]
            healLongL = (matrix_world @ bone.tail) - (matrix_world @ bone.head)
            bone.head = matrix_world_inv @ Vector(key_point_location["左腳跟"])
            bone.tail = matrix_world_inv @ ((matrix_world @ bone.head) + healLongL)
            
            bone = armature.data.edit_bones["pelvis.R"]
            bone.head = matrix_world_inv @ Vector(key_point_location["骨盆"])
            bone.tail = matrix_world_inv @ Vector(key_point_location["右大腿跟"])
            bone = armature.data.edit_bones["thigh.R"]#右腿1
            bone.head = matrix_world_inv @ (Vector(key_point_location["右大腿跟"])) #球 右腳1
            bone.tail = matrix_world_inv @ (Vector(key_point_location["右膝"])) #球 右腳2
            bone = armature.data.edit_bones["shin.R"]#右腿2
            bone.tail = matrix_world_inv @ (Vector(key_point_location["右腳踝"])) #球 右腳3
            bone = armature.data.edit_bones["foot.R"]#右腳
            bone.tail = matrix_world_inv @ (Vector(key_point_location["右腳底"])) #球 右腳4
            bone = armature.data.edit_bones["toe.R"]#右腳趾
            bone.head = matrix_world_inv @ Vector(key_point_location["右腳底"])
            bone.tail = matrix_world_inv @ (Vector(key_point_location["右腳趾尖"]))#球 右腳4
            bone = armature.data.edit_bones["heel.02.R"]
            healLongR = (matrix_world @ bone.tail) - (matrix_world @ bone.head)
            bone.head = matrix_world_inv @ Vector(key_point_location["右腳跟"])
            bone.tail = matrix_world_inv @ ((matrix_world @ bone.head) + healLongR)

            # 重新計算骨頭滾動
            bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Y') # 對齊到 +Y 方向

            bpy.ops.object.mode_set(mode='OBJECT') # 離開骨頭的編輯模式
        else:
            print("目前的活動物體不是骨架")
        return {'FINISHED'}

class SimpleOperator4(bpy.types.Operator):
    bl_idname = "wm.my_operator4"
    bl_label = "蒙皮輸出"

    def execute(self, context):
        bpy.types.Scene.enable_button4 = bpy.props.BoolProperty(name="禁用按鈕 4", default=False)
        
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.transform.translate(value=(-Vector(key_point_location["骨盆"]).x, 0, 0), orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)
        
        myObject = bpy.context.scene.objects[file_name]  # Replace with statement to select object
        myArmature =  bpy.context.scene.objects['metarig'] # Replace with statement to select the correct armature

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.transform.resize(value=(5, 5, 5), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', 
                                proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, 
                                snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, 
                                use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)

        # Assumes that nothing else is selected at the moment
        myObject.select_set(True)  # Must select the object first
        myArmature.select_set(True) # Then select the armature
        bpy.ops.object.parent_set(type='ARMATURE_AUTO') # then parent
        
        bpy.ops.transform.resize(value=(0.2, 0.2, 0.2), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), 
                                orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', 
                                proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, 
                                snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, 
                                use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False)

        
        # 設定匯出路徑和檔案名稱
        export_path = "../ModelOutput/people.fbx"
        # 設定匯出設定
        bpy.ops.export_scene.fbx(
            filepath=export_path,
            #check_existing=True,        # 檢查檔案是否已存在
            #axis_forward='Y',           # 設定前方軸為Y軸
            #axis_up='Z',                # 設定上方軸為Z軸
            use_selection=True,         # 匯出所選的物體
            #object_types={'MESH', 'ARMATURE', 'EMPTY'},  # 要匯出的物體類型
            #use_mesh_modifiers=True,    # 是否應用網格修改器
            #use_mesh_edges=False,       # 是否匯出網格邊線
            #use_tspace=True,            # 是否匯出切線/副法線
            add_leaf_bones=True,       # 是否添加末梢骨頭
            #use_anim=True,              # 是否匯出動畫
            #use_anim_action_all=True,   # 是否匯出所有動畫動作
            #use_default_take=True,      # 是否匯出默認動畫動作
            #use_anim_optimize=True,     # 是否優化動畫
            #anim_optimize_precision=6,  # 優化精度
            #path_mode='AUTO',           # 路徑模式
            #embed_textures=False,       # 是否嵌入貼圖
            #batch_mode='OFF',           # 批次模式
            #use_batch_own_dir=True,     # 是否使用批次模式的自己的目錄
            #use_metadata=True,          # 是否匯出元數據
            #use_custom_props=True,      # 是否匯出自定義屬性
        )
        return {'FINISHED'}

def register():
    bpy.utils.register_class(SimplePanel)
    bpy.types.Scene.enable_button1 = bpy.props.BoolProperty(name="啟用按鈕 1", default=True)
    bpy.types.Scene.enable_button2 = bpy.props.BoolProperty(name="啟用按鈕 2", default=True)
    bpy.utils.register_class(SimpleOperator1)
    bpy.utils.register_class(SimpleOperator2)
    bpy.utils.register_class(SimpleOperator3)
    bpy.utils.register_class(SimpleOperator4)

def unregister():
    bpy.utils.unregister_class(SimplePanel)
    bpy.utils.unregister_class(SimpleOperator1)
    bpy.utils.unregister_class(SimpleOperator2)
    bpy.utils.unregister_class(SimpleOperator3)
    bpy.utils.unregister_class(SimpleOperator4)
    del bpy.types.Scene.enable_button1
    del bpy.types.Scene.enable_button2
    del bpy.types.Scene.enable_button3
    del bpy.types.Scene.enable_button4
    
register()