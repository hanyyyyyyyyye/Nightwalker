import sys
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer.mesh.shading import phong_shading
import pytorch3d as p3d
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from pytorch3d.renderer import (
    cameras,
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
    TexturesUV
)
import matplotlib.pyplot as plt
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision
import math
sys.path.append("/home/Newdisk/yanyunjie/code_practics/patch/Adversarial_camou-main/tutorials/")
sys.path.append("/home/Newdisk/yanyunjie/code_practics/infraCam/models/yolov5-master")
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
# sys.path.append("/home/Newdisk/yanyunjie/code_practics/SigPri/yolo3-pytorch-master")
from plot_image_grid import image_grid
unloader=transforms.ToPILImage()
# from yolo import YOLO
# def cal_det_loss(adv_batch,gt):
#     yolo = YOLO(device)
#     model = yolo.net.to(device).eval()
#     image_shape = adv_batch.shape[2:]
#     outputs = model(adv_batch)
#     outputs = yolo.bbox_util.decode_box(outputs)
#     results = yolo.bbox_util.non_max_suppression(torch.cat(outputs, 1), yolo.num_classes, yolo.input_shape,
#                                                  image_shape, yolo.letterbox_image, conf_thres=yolo.confidence,
#                                                  nms_thres=yolo.nms_iou)
#     iou_thread = 0.01
#     num = 0
#     max_probs = []
#     det_loss = []
#     for i, res in enumerate(results):
#         bbox = torch.stack([res[..., 0], res[..., 2], res[..., 3], res[..., 4]], dim=-1)
#         ious = torchvision.ops.box_iou(bbox, gt[i].unsqueeze(0)).squeeze(-1)
#         mask = ious.ge(iou_thread)
#         mask = mask.logical_and(res[..., 6] == 0)
#         ious = ious[mask]
#         scores = res[..., 4][mask]
#         if len(ious) != 0:
#             _, ids = torch.max(ious, dim=0)  # get the bbox w/ biggest iou compared to gt
#             det_loss.append(scores[ids])
#             max_probs.append(scores[ids])
#             num += 1
#     if num < 1:
#         raise RuntimeError()
#     else:
#         det_loss = torch.stack(det_loss).mean()
#         max_probs = torch.stack(max_probs)
#         return det_loss, max_probs
def get_adv_batch(image_pred,batch_size):
    mask = (image_pred.permute(0, 3, 1, 2)[:, -1:, ...] > 0).to(image_pred.permute(0, 3, 1, 2)).repeat(1, 3, 1, 1)
    adv_patch = image_pred.permute(0, 3, 1, 2)[:, :-1, ...]

    min_scale = 0.2
    max_scale = 0.5
    Ho = 640
    Wo = 640
    translation_x = 0.8
    translation_y = 1.0
    B = image_pred.shape[0]
    scale = adv_patch.new(size=[B]).uniform_(min_scale, max_scale).exp()
    mesh_bord = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
    mesh_bord = mesh_bord / mesh_bord.new([Ho, Wo, Ho, Wo]) * 2 - 1
    #         mesh_bord = mesh_bord / scale
    pos_param = mesh_bord + mesh_bord.new([1, 1, -1, -1]) * scale.unsqueeze(-1)
    tymin, txmin, tymax, txmax = pos_param.unbind(-1)

    xdiff = (-txmax + txmin).clamp(min=0)
    xmiddle = (txmax + txmin) / 2
    ydiff = (-tymax + tymin).clamp(min=0)
    ymiddle = (tymax + tymin) / 2

    tx = txmin.new(txmin.shape).uniform_(-0.5, 0.5) * xdiff * translation_x + xmiddle
    ty = tymin.new(tymin.shape).uniform_(-0.5, 0.5) * ydiff * translation_y + ymiddle

    theta = adv_patch.new_zeros(B, 2, 3)
    theta[:, 0, 0] = scale
    theta[:, 0, 1] = 0
    theta[:, 1, 0] = 0
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, adv_patch.shape)
    adv_batch_ori = F.grid_sample(adv_patch, grid, padding_mode='zeros')
    mask = F.grid_sample(mask, grid, padding_mode='zeros')
    image_batch,bg_list = generate_background_batch(batch_size,image_pred.shape[1:3])
    image_batch=image_batch.to(device)
    adv_batch = adv_batch_ori * mask + image_batch * (1 - mask)
    gt = torch.stack([torch.cat([m[0].nonzero().min(0).values, m[0].nonzero().max(0).values]) for m in mask])
    gt = gt[:, [1, 0, 3, 2]].unbind(0)
    return adv_batch,gt,adv_batch_ori,bg_list
def generate_background_batch(batch_size,bg_shape):
    ToTensor=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(bg_shape),
    ])
    bg_path = "/home/Newdisk/yanyunjie/code_practics/infraCam/dataset/FLIR/VOCdevkit/VOC2007/JPEGImages"
    background_list=[]
    with open("/home/Newdisk/yanyunjie/code_practics/infraCam/dataset/name_list.txt","r") as f:
        lines=f.readlines()
        for line in lines:
            background_list.append(line.split("\n")[0]+".jpg")
    bg_list=random.sample(background_list,batch_size)
    for i,img_path in enumerate(bg_list):
        image=Image.open(os.path.join(bg_path,img_path))
        image_tensor=ToTensor(image)
        if i==0:
            img_batch=image_tensor.unsqueeze(0)
        else:
            img_batch=torch.cat((img_batch,image_tensor.unsqueeze(0)))
    return img_batch,bg_list




def visilize_batch(image_batch):
    image_grid(image_batch.permute(0, 2, 3, 1).cpu().numpy(), rows=4, cols=5, rgb=True)
    plt.show()
def visilize(images):
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.show()
def join_meshes(meshes, join_maps=None):
    verts = []
    faces = []
    verts_uvs = []
    faces_uvs = []
    maps = []
    for mesh in meshes:
        verts.append(mesh.verts_packed())
        faces.append(mesh.faces_packed())
        maps.append(mesh.textures.maps_list()[0])
        verts_uvs.append(mesh.textures.verts_uvs_list()[0])
        faces_uvs.append(mesh.textures.faces_uvs_list()[0])

    w = 0
    h = 0
    pos = []
    for m in maps:
        if m.shape[0] > w:
            w = m.shape[0]
        h = h + m.shape[1]

    hi = 0
    v_num = 0
    vuv_num = 0
    for i in range(len(meshes)):
        verts_uvs[i] = torch.stack(
            [(verts_uvs[i][:, 0] * maps[i].shape[1] + hi) / h, verts_uvs[i][:, 1] * maps[i].shape[0] / w], -1)
        hi = hi + maps[i].shape[1]

        faces[i] = faces[i] + v_num
        v_num += len(verts[i])

        faces_uvs[i] = faces_uvs[i] + vuv_num
        vuv_num += len(verts_uvs[i])

    if join_maps is None:
        maps = [F.pad(m, (0, 0, 0, 0, w - m.shape[0], 0)) for m in maps]
        join_maps = [torch.cat(maps, 1)]

    verts = [torch.cat(verts)]
    faces = [torch.cat(faces)]
    verts_uvs = [torch.cat(verts_uvs)]
    faces_uvs = [torch.cat(faces_uvs)]


    textures = p3d.renderer.mesh.textures.TexturesUV(join_maps, faces_uvs, verts_uvs)
    return Meshes(verts=verts, faces=faces, textures=textures)
def fragments_reprojection(fragments, start, end, mesh, locations, infos):
    """
    Only modify the fragments.pix_to_face and fragments.bary_coords
    One need to use MyHardPhongShader, otherwise the shader renders black area on the closest face
    """
    grids_bc_kernels = infos['grids_bc_kernels']
    grids_index = infos['grids_index']
    pix_to_face = fragments.pix_to_face
    bary_coords = fragments.bary_coords
    pix_selected = (pix_to_face < end).logical_and(pix_to_face >= start)

    pf = pix_to_face[pix_selected] - start
    bc = bary_coords[pix_selected]

    # get new coords after tps
    faces_uvs = mesh.textures.faces_uvs_list()[0]
    faces_locs = locations[faces_uvs.view(-1)].view(-1, 3, 2)
    coords = (faces_locs[pf] * bc.unsqueeze(-1)).sum(1)

    # compute coords bins
    indexes = ((coords + infos['max_range']) / infos['bin_size']).round().clamp(0, infos['bin_num']).long()
    indexes = indexes[:, 0] * (infos['bin_num'] + 1) + indexes[:, 1]
    bc_true = F.pad(coords, [0, 1], value=1.0).view(-1, 1, 1, 3).matmul(grids_bc_kernels[indexes]).squeeze(-2)

    indicator = (bc_true[..., 0] >= 0).logical_and((bc_true[..., 1] >= 0).logical_and(bc_true[..., 2] >= 0))
    inds = indicator.max(1)[1]

    pf_new = grids_index[indexes].gather(-1, inds.unsqueeze(-1)).squeeze(-1)
    bc_new = bc_true.gather(1, inds.view(-1, 1, 1).expand(-1, -1, 3)).squeeze(1)

    fragments.pix_to_face[pix_selected] = pf_new + start * (pf_new >= 0)
    fragments.bary_coords[pix_selected] = bc_new
    return fragments


def my_hard_rgb_blend(
    colors: torch.Tensor, fragments, blend_params) -> torch.Tensor:
    """
    Modification of pytorch3d.renderer.blending.hard_rgb_blend
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # my modification, to fit the flexible pix_to_face
    inds = (fragments.pix_to_face >= 0).max(-1)[1][..., None]
    pix_to_face = fragments.pix_to_face.gather(3, inds)
    colors = colors.gather(3, inds[..., None].expand(-1, -1, -1, -1, colors.shape[-1]))

    # Mask for the background.
    is_background = pix_to_face[..., 0] < 0  # (N, H, W)

    background_color_ = blend_params.background_color
    if isinstance(background_color_, torch.Tensor):
        background_color = background_color_.to(device)
    else:
        background_color = colors.new_tensor(background_color_)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    num_background_pixels = is_background.sum()

    # Set background color.
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, 3)

    # Concat with the alpha channel.
    alpha = (~is_background).type_as(pixel_colors)[..., None]

    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)


class MyHardPhongShader(HardPhongShader):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardPhongShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardPhongShader"
            raise ValueError(msg)

        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = my_hard_rgb_blend(colors, fragments, blend_params)
        return images


def view_mesh_wrapped(mesh_list, locations_list=None, infos_list=None, offset_verts=None, cameras=(0, 0, 0), lights=None, up=(0, 1, 0), image_size=512, device=None, fov=60, background=None, **kwargs):
    mesh_join = join_meshes(mesh_list)
    num_faces = len(mesh_join.faces_packed())
    if device is None:
        device = mesh_join.device
    if isinstance(cameras, list) or isinstance(cameras, tuple):
        R, T = look_at_view_transform(*cameras, up=(up,))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)
        # cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    if len(mesh_join) == 1 and len(cameras) > 1:
        mesh_join = mesh_join.extend(len(cameras))
    elif len(mesh_join) != len(cameras):
        print('mesh num %d and camera %d num mis-match' % (len(mesh_join), len(cameras)))
        raise ValueError

    if offset_verts is not None:
        mesh_join.offset_verts_(offset_verts - mesh_join.verts_packed())

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=kwargs.get('blur_radius', 0.0),
        faces_per_pixel=kwargs.get('faces_per_pixel', 1),
        bin_size=kwargs.get('bin_size', None),
        max_faces_per_bin=kwargs.get('max_faces_per_bin', None),
    )

    if lights is None:
        lights = AmbientLights(device=device)
    #     lights = PointLights(device=device, location=[light_loc])

    blend_params = BlendParams(1e-4, 1e-4, background) if background is not None else None

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    shader = MyHardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        blend_params=blend_params
    )

    fragments = rasterizer(mesh_join)

    if locations_list is not None:
        start = 0
        for mesh, locations, infos in zip(mesh_list, locations_list, infos_list):
            end = start + mesh.faces_list()[0].shape[0]
            if locations is not None:
                for i in range(len(mesh_join)):
                    fragments = fragments_reprojection(fragments, start + i*num_faces, end + i*num_faces, mesh, locations[i], infos)
            start = end

    images = shader(fragments, mesh_join)
    return images
def voronoi_diagram(point_number,coordinates,fig_size_h, fig_size_w,color_list,device):
    mask1=torch.ones_like(color_list).to(device)*0.8
    mask0 = torch.ones_like(color_list).to(device)*0.2
    mean=torch.unique(color_list)[int(len(torch.unique(color_list))/2)]
    # color=color_list.clone()
    color_list=torch.where(color_list>mean,mask1,color_list)
    color_list=torch.where(color_list<=mean,mask0,color_list).repeat(1,3)
    coordinates = coordinates.expand(point_number.shape[0], -1, -1, -1).permute(1, 2, 0, 3)
    circle0 = point_number[..., 0] * fig_size_h
    circle1 = point_number[..., 1] * fig_size_w

    # 构成坐标
    circles = torch.stack([circle0, circle1], dim=-1)  # [颜色数量,选择点的数量，二维坐标]
    dist = torch.norm(coordinates - circles[:, :2], dim=-1)
    index = torch.argmin(dist, dim=-1)
    result = color_list[index]
    return result
def get_vd(num_colors,tshirt_point_num, trouser_point_num,color_list, color_list_t):
    fig_size_H = 340
    fig_size_W = 864

    fig_size_H_t = 484
    fig_size_W_t = 700
    resolution = 4
    h, w, h_t, w_t = int(fig_size_H / resolution), int(fig_size_W / resolution), int(
            fig_size_H_t / resolution), int(fig_size_W_t / resolution)
    coordinates = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).to(device)
    coordinates_t = torch.stack(torch.meshgrid(torch.arange(h_t), torch.arange(w_t)), -1).to(device)

    k = 3
    k2 = k * k
    camouflage_kernel = nn.Conv2d(num_colors, num_colors, k, 1, int(k / 2)).to(device)
    camouflage_kernel.weight.data.fill_(0)
    camouflage_kernel.bias.data.fill_(0)
    for i in range(num_colors):
        camouflage_kernel.weight[i, i, :, :].data.fill_(1 / k2)

    expand_kernel = nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)



    tex = voronoi_diagram(tshirt_point_num, coordinates, h, w, color_list,
                          device).unsqueeze(0)
    tex_trouser = voronoi_diagram(trouser_point_num, coordinates_t, h_t, w_t,color_list_t,
                                  device).unsqueeze(0)
    return tex,tex_trouser
def get_map_kernel(locations, faces_uvs_all, use_grids=True, bin_num=50, padded_default=True, batch_size=2000):
    """
    get the barycentric kernels for a map
    """
    faces_locs = locations[faces_uvs_all.view(-1)].view(-1, 3, 2)
    bc_kernel = torch.inverse(F.pad(faces_locs, [0, 1], value=1.0))
    if not use_grids:
        return bc_kernel

    # use grids to save memory
    max_range = locations.abs().max()
    bin_size = 2 * max_range / bin_num
    bin_range = bin_size / np.sqrt(2)
    # compute the grids coordinates
    grids = torch.meshgrid(torch.linspace(-max_range, max_range, bin_num + 1),
                           torch.linspace(-max_range, max_range, bin_num + 1))
    grids = torch.stack(grids, -1).view(-1, 2).to(locations)

    # split faces in batches
    faces_uvs_list = faces_uvs_all.split(batch_size, 0)
    collect = []
    counts = []
    for fi, faces_uvs in enumerate(faces_uvs_list):
        faces_locs = locations[faces_uvs.view(-1)].view(-1, 3, 2)
        # bc_kernel = torch.inverse(F.pad(faces_locs, [0, 1], value=1.0))

        # compute the distance to triangles
        v1 = faces_locs.view(1, -1, 3, 2) - grids.view(-1, 1, 1, 2)
        v2 = grids.view(-1, 1, 1, 2) - faces_locs.roll(1, 1).view(1, -1, 3, 2)
        v3 = faces_locs.roll(1, 1).view(1, -1, 3, 2) - faces_locs.view(1, -1, 3, 2)
        lambda1 = - (v2 * v3).sum(-1, keepdim=True)
        lambda2 = (v1 * v3).sum(-1, keepdim=True)
        perp = (lambda1 * v1 + lambda2 * v2) / (v3 * v3).sum(-1, keepdim=True)
        perp_norm = perp.norm(2, -1)
        points_min = v1.norm(2, -1).minimum(v2.norm(2, -1))
        indicator = (lambda1 >= 0).logical_and(lambda2 <= 0).squeeze(-1)
        dis_to_tri = torch.where(indicator, perp_norm, points_min).min(-1)[0]

        area1 = (v1 * v2.flip(-1)).sum(-1).abs().sum(-1)
        area2 = ((faces_locs[:, 0] - faces_locs[:, 1]) * (faces_locs[:, 0] - faces_locs[:, 2]).flip(-1)).sum(
            -1).abs().unsqueeze(0)
        dis_to_tri = torch.where(area1 > area2, dis_to_tri, dis_to_tri.new([0.0]))  # Ng * B
        # dis_to_tri = perp.norm(2, -1).min(-1)[0]

        in_tri = dis_to_tri <= bin_range
        collect_i = in_tri.nonzero()
        collect_i[:, 1] += fi * batch_size
        collect.append(collect_i)

        counts_i = torch.count_nonzero(in_tri, dim=1)
        counts.append(counts_i)

        # max_num = in_tri.sum(1).max().item()
        # print('max range is %.3f, max number of the bins is %d' % (max_range, max_num))
        #
        # if padded_default:
        #     grids_indicator, i = [F.pad(x, [1, 0]) for x in in_tri.long().topk(max_num)]
        # else:
        #     grids_indicator, i = in_tri.long().topk(max_num)
        #
        # # grids_index = grids_indicator * i + (1 - grids_indicator) * -1
        # grids_index = torch.where(grids_indicator.bool(), i, -1)
        #
        # grids_bc_kernels = bc_kernel[grids_index]
        # kernel_fake = grids_bc_kernels.new(
        #     [[max_range + 1, max_range + 1], [max_range + 2, max_range + 1], [max_range + 1, max_range + 2]])
        # kernel_fake = torch.inverse(F.pad(kernel_fake, [0, 1], value=1.0))
        # grids_bc_kernels = grids_bc_kernels * grids_indicator.unsqueeze(-1).unsqueeze(-1) - kernel_fake * (
        #             1 - grids_indicator.unsqueeze(-1).unsqueeze(-1))
    collect = torch.cat(collect, 0)
    collect = collect[collect[:, 0].argsort()]
    counts = torch.stack(counts, -1).sum(-1)
    max_num = counts.max().item()
    print('max range is %.3f, max number of the bins is %d' % (max_range, max_num))

    ids = collect[:, 0]
    values = collect[:, 1]
    num_ids = F.pad(counts.unsqueeze(1).expand(-1, len(ids)).triu().sum(0)[:-1], [1, 0], value=0)
    ids_per_bin = torch.arange(len(collect), device=locations.device) - num_ids[ids]

    grids_index_tt = ids.new_zeros(size=(len(grids), max_num)) - 1
    grids_index_tt.index_put_([ids, ids_per_bin], values)
    # return collect, counts

    kernel_fake = locations.new([[max_range + 1, max_range + 1], [max_range + 2, max_range + 1], [max_range + 1, max_range + 2]])
    kernel_fake = torch.inverse(F.pad(kernel_fake, [0, 1], value=1.0))

    if padded_default:
        grids_index_tt = F.pad(grids_index_tt, [1, 0], value=-1)

    grids_bc_kernels_tt = torch.where(grids_index_tt[..., None, None].expand(-1, -1, 3, 3) >= 0, bc_kernel[grids_index_tt], kernel_fake[None, None, :])

    infos = {
        'grids_bc_kernels': grids_bc_kernels_tt,
        # 'grids_indicator': grids_indicator_tt,
        'grids_index': grids_index_tt,
        'bin_num': bin_num,
        'max_range': max_range,
        'bin_size': bin_size,
        'bn_tt': len(grids_index_tt),
    }
    return infos


#运行环境配置
sys.path.append("/home/Newdisk/yanyunjie/code_practics/patch/Adversarial_camou-main")
device = torch.device("cuda:5")
torch.cuda.set_device(device)
#载入3个mesh
def join_mesh_render(azim,elev,lights,img_shape,num_colors,tshirt_point_num, trouser_point_num,color_list, color_list_t):

    obj_filename_man = "/home/Newdisk/yanyunjie/code_practics/patch/Adversarial_camou-main/data/Archive/Man_join/man.obj"
    obj_filename_tshirt = "/home/Newdisk/yanyunjie/code_practics/patch/Adversarial_camou-main/data/Archive/tshirt_join/tshirt.obj"
    obj_filename_trouser = "/home/Newdisk/yanyunjie/code_practics/patch/Adversarial_camou-main/data/Archive/trouser_join/trouser.obj"
    tex,tex_trouser=get_vd(num_colors,tshirt_point_num, trouser_point_num,color_list, color_list_t)#
    # tex=torch.ones_like(tex).to(device).mul(0.8)
    # tex_trouser=torch.ones_like(tex_trouser).to(device).mul(0.8)
    mesh_man = load_objs_as_meshes([obj_filename_man], device=device)
    mesh_tshirt = load_objs_as_meshes([obj_filename_tshirt], device=device)
    mesh_trouser = load_objs_as_meshes([obj_filename_trouser], device=device)

    faces = mesh_tshirt.textures.faces_uvs_padded()
    verts_uv = mesh_tshirt.textures.verts_uvs_padded()
    faces_uvs_tshirt = mesh_tshirt.textures.faces_uvs_list()[0]

    faces_trouser = mesh_trouser.textures.faces_uvs_padded()
    verts_uv_trouser = mesh_trouser.textures.verts_uvs_padded()
    faces_uvs_trouser = mesh_trouser.textures.faces_uvs_list()[0]
    mesh_man.textures._maps_padded=torch.ones_like(mesh_man.textures.maps_padded()).to(device).mul(0.9)
    mesh_tshirt.textures = TexturesUV(maps=tex, faces_uvs=faces, verts_uvs=verts_uv)
    mesh_trouser.textures = TexturesUV(maps=tex_trouser, faces_uvs=faces_trouser,
                                       verts_uvs=verts_uv_trouser)
    locations_tshirt_ori = torch.load(
        os.path.join("/home/Newdisk/yanyunjie/code_practics/patch/Adversarial_camou-main/data/Archive/tshirt_join/projections/part_all_2p5.pt"), map_location='cpu').to(
        device)
    infos_tshirt = get_map_kernel(locations_tshirt_ori, faces_uvs_tshirt)

    locations_trouser_ori = torch.load(
        os.path.join("/home/Newdisk/yanyunjie/code_practics/patch/Adversarial_camou-main/data/Archive/trouser_join/projections/part_all_off3p4.pt"), map_location='cpu').to(device)
    infos_trouser = get_map_kernel(locations_trouser_ori, faces_uvs_trouser)
    R, T = look_at_view_transform(dist=2.5, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=45)
    source_coordinate=None
    image_pred=view_mesh_wrapped([mesh_man, mesh_tshirt, mesh_trouser],
                      [None, None, None],
                      [None, infos_tshirt, infos_trouser], source_coordinate,
                      cameras=cameras, lights=lights, image_size=img_shape, fov=45,
                      max_faces_per_bin=30000, faces_per_pixel=3).clamp(0,1)
    return image_pred
    # visilize_batch(adv_batch)
def sample_lights(r=None):
    if r is None:
        r = np.random.rand()
    theta = np.random.rand() * 2 * math.pi
    if r < 0.33:
        lights = AmbientLights(device=device)
    elif r < 0.67:
        lights = DirectionalLights(device=device, direction=[[np.sin(theta), 0.0, np.cos(theta)]])
    else:
        lights = PointLights(device=device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]])
    return  PointLights(device=device, location=[[np.sin(theta) * 3, 0.0, np.cos(theta) * 3]])
def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
def detect_yolov5(model,img):
    conf_thres = 0.4  # confidence threshold
    iou_thres = 0.45
    max_nms=30000
    max_wh = 7680
    classes = None
    agnostic_nms = False
    max_det = 1000
    pred, _ = model(img)
    xc=pred[...,4]>conf_thres
    output=[]
    for xi,x in enumerate(pred):
        print(xc[xi].shape)
        x_f=x[xc[xi]]
        if not x_f.shape[0]:
            continue
        x_f[:,5:]*=x_f[:,4:5]
        box = xywh2xyxy(x_f[:, :4])
        conf, j = x_f[:, 5:].max(1, keepdim=True)
        out=torch.cat((box, conf, j.float()),dim=1)[conf.view(-1) > conf_thres]
        out=out[out[:,4].argsort(descending=True)[:max_nms]]
        c = out[:, 5:6]*max_wh
        boxes, scores=out[:,:4]+c,out[:,4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if len(out[i])>0:
            output.append(out[i])

    return output
def cal_det_loss(pred,gt):
    conf_thres = 0.4  # confidence threshold
    iou_thres = 0.45
    max_nms = 30000
    max_wh = 7680
    xc = pred[..., 4] > conf_thres
    det_loss = []
    for xi, x in enumerate(pred):
        x_f = x[xc[xi]]
        bbox= (xywh2xyxy(x_f[:, :4])+x_f[:,5:6]*max_wh).clamp(0,640)
        ious=torchvision.ops.box_iou(bbox, gt[xi].unsqueeze(0)).squeeze(-1)
        mask = ious.ge(iou_thres)
        mask=mask.logical_and(x_f[:, 5:].max(1)[1] == 0)
        scores=x_f[:,4][mask]
        ious=ious[mask]
        if len(ious)>0:
            _, ids = torch.max(ious, dim=0)  # get the bbox w/ biggest iou compared to gt
            det_loss.append(scores[ids])
    if len(det_loss)!=0:
        det_loss = torch.stack(det_loss).mean()
    else:
        det_loss=None
    return det_loss
def smooth(img):
    mask1 = (img[:, :, 1:, :-1] != 0) * 1
    mask2 = (img[:, :, :-1, :-1] != 0) * 1
    maska = (mask1 == mask2) * 1
    mask3 = (img[:, :, :-1, 1:] != 0) * 1
    mask4 = (img[:, :, :-1, :-1] != 0) * 1
    maskb = (mask3 == mask4) * 1
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2) * maska
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2) * maskb

    return torch.sum(s1 + s2)



if __name__=="__main__":
    torch.autograd.set_detect_anomaly(True)
    num_colors = 1
    num_points_tshirt = 500
    num_points_trouser = 500
    tshirt_point = torch.rand([num_colors, num_points_tshirt, 3], requires_grad=True, device=device)
    tshirt_point_num = torch.rand([tshirt_point.shape[1], 2], requires_grad=True, device=device)
    trouser_point = torch.rand([num_colors, num_points_trouser, 3], requires_grad=True, device=device)
    trouser_point_num = torch.rand([trouser_point.shape[1], 2], requires_grad=True, device=device)
    color_list = torch.rand([num_points_tshirt, 1], requires_grad=True, device=device)
    color_list_t = torch.rand([num_points_trouser, 1], requires_grad=True, device=device)
    optimizer = torch.optim.Adam([tshirt_point_num, trouser_point_num], lr=0.001)
    optimizer_seed = torch.optim.Adam([color_list, color_list_t], lr=0.1)


    epoch=100
    for e in range(epoch):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        optimizer_seed.zero_grad()
        batch_size = 4
        azim = (torch.zeros(batch_size).uniform_() - 0.5) * 360
        elev = 10 + 8 * torch.zeros(batch_size).uniform_(-1, 1)
        lights = sample_lights()
        img_size=640
        image_pred=join_mesh_render(azim,elev, lights,img_size,num_colors,tshirt_point_num, trouser_point_num,color_list, color_list_t)
        adv_batch, gt,adv_batch_ori,bg_list = get_adv_batch(image_pred, batch_size)
        weights_path = "/home/Newdisk/yanyunjie/code_practics/infraCam/models/yolov5-master/runs/train/exp4/weights/best.pt"
        model = DetectMultiBackend(weights_path, device=device)
        model.eval()
        pred, _ = model(adv_batch)
        # conf_thres = 0.4  # confidence threshold
        # iou_thres = 0.45
        # classes = None
        # agnostic_nms = False
        # max_det = 1000
        # pred_result = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print(pred_result)
        det_loss=cal_det_loss(pred,gt)
        print(det_loss)
        if det_loss==None:
            save_batch=adv_batch.clone()
            for i in range(len(save_batch)):
                image=save_batch[i]
                image_o=adv_batch_ori[i]
                img=unloader(image)
                img_o=unloader(image_o)
                img.save(os.path.join("/home/Newdisk/yanyunjie/code_practics/infraCam/mymethod/result",bg_list[i]))
                img_o.save(os.path.join("/home/Newdisk/yanyunjie/code_practics/infraCam/mymethod/result",bg_list[i].split(".jpg")[0]+"_ori"+".jpg"))
            continue

        # det_loss=pred[0][...,4].sum()
        tv_loss=smooth(adv_batch_ori)*0.000001
        loss = det_loss + tv_loss
        # loss=tv_loss
        # loss.requires_grad_(True)
        print("loss:",loss,"det_loss:",det_loss,"tv_loss:",tv_loss)
        with torch.autograd.detect_anomaly():
            loss.backward()
        print(color_list_t.grad)
        optimizer.step()
        optimizer_seed.step()
        tshirt_point.data = tshirt_point.data.clamp(0, 1)
        color_list.data = color_list.data.clamp(0, 1)
        trouser_point.data = trouser_point.data.clamp(0, 1)
        color_list_t.data = color_list_t.data.clamp(0, 1)






