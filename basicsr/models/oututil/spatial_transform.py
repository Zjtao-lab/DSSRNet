import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('/home/thinkstation03/zjt/NAFNet-ALL/outpackage')
# import package_name
from kornia.geometry.transform import rotate
# from .imgwarp import get_affine_matrix2d, get_projective_transform, get_rotation_matrix2d, warp_affine, warp_affine3d

# github https://github.com/kornia/kornia/tree/master/kornia/geometry/transform

# def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
#     """Compute the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
#     if not 2 <= len(tensor.shape) <= 4:
#         raise AssertionError(f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}.")
#     height, width = tensor.shape[-2:]
#     center_x: float = float(width - 1) / 2
#     center_y: float = float(height - 1) / 2
#     center: torch.Tensor = torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)
#     return center

# def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
#     """Compute a pure affine rotation matrix."""
#     scale: torch.Tensor = torch.ones_like(center)
#     matrix: torch.Tensor = get_rotation_matrix2d(center, angle, scale)
#     return matrix

# def affine(
#     tensor: torch.Tensor,
#     matrix: torch.Tensor,
#     mode: str = 'bilinear',
#     padding_mode: str = 'zeros',
#     align_corners: bool = True,
# ) -> torch.Tensor:
#     r"""Apply an affine transformation to the image.

#     .. image:: _static/img/warp_affine.png

#     Args:
#         tensor: The image tensor to be warped in shapes of
#             :math:`(H, W)`, :math:`(D, H, W)` and :math:`(B, C, H, W)`.
#         matrix: The 2x3 affine transformation matrix.
#         mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
#         padding_mode: padding mode for outside grid values
#           ``'zeros'`` | ``'border'`` | ``'reflection'``.
#         align_corners: interpolation flag.

#     Returns:
#         The warped image with the same shape as the input.

#     Example:
#         >>> img = torch.rand(1, 2, 3, 5)
#         >>> aff = torch.eye(2, 3)[None]
#         >>> out = affine(img, aff)
#         >>> print(out.shape)
#         torch.Size([1, 2, 3, 5])
#     """
#     # warping needs data in the shape of BCHW
#     is_unbatched: bool = tensor.ndimension() == 3
#     if is_unbatched:
#         tensor = torch.unsqueeze(tensor, dim=0)

#     # we enforce broadcasting since by default grid_sample it does not
#     # give support for that
#     matrix = matrix.expand(tensor.shape[0], -1, -1)

#     # warp the input tensor
#     height: int = tensor.shape[-2]
#     width: int = tensor.shape[-1]
#     warped: torch.Tensor = warp_affine(tensor, matrix, (height, width), mode, padding_mode, align_corners)

#     # return in the original shape
#     if is_unbatched:
#         warped = torch.squeeze(warped, dim=0)

#     return warped

# def rotate(
#     tensor: torch.Tensor,
#     angle: torch.Tensor,
#     center: Union[None, torch.Tensor] = None,
#     mode: str = 'bilinear',
#     padding_mode: str = 'zeros',
#     align_corners: bool = True,
# ) -> torch.Tensor:
#     r"""Rotate the tensor anti-clockwise about the center.

#     .. image:: _static/img/rotate.png

#     Args:
#         tensor: The image tensor to be warped in shapes of :math:`(B, C, H, W)`.
#         angle: The angle through which to rotate. The tensor
#           must have a shape of (B), where B is batch size.
#         center: The center through which to rotate. The tensor
#           must have a shape of (B, 2), where B is batch size and last
#           dimension contains cx and cy.
#         mode: interpolation mode to calculate output values
#           ``'bilinear'`` | ``'nearest'``.
#         padding_mode: padding mode for outside grid values
#           ``'zeros'`` | ``'border'`` | ``'reflection'``.
#         align_corners: interpolation flag.

#     Returns:
#         The rotated tensor with shape as input.

#     .. note::
#        See a working example `here <https://kornia.github.io/tutorials/nbs/rotate_affine.html>`__.

#     Example:
#         >>> img = torch.rand(1, 3, 4, 4)
#         >>> angle = torch.tensor([90.])
#         >>> out = rotate(img, angle)
#         >>> print(out.shape)
#         torch.Size([1, 3, 4, 4])
#     """
#     if not isinstance(tensor, torch.Tensor):
#         raise TypeError(f"Input tensor type is not a torch.Tensor. Got {type(tensor)}")

#     if not isinstance(angle, torch.Tensor):
#         raise TypeError(f"Input angle type is not a torch.Tensor. Got {type(angle)}")

#     if center is not None and not isinstance(center, torch.Tensor):
#         raise TypeError(f"Input center type is not a torch.Tensor. Got {type(center)}")

#     if len(tensor.shape) not in (3, 4):
#         raise ValueError(f"Invalid tensor shape, we expect CxHxW or BxCxHxW. Got: {tensor.shape}")

#     # compute the rotation center
#     if center is None:
#         center = _compute_tensor_center(tensor)

#     # compute the rotation matrix
#     # TODO: add broadcasting to get_rotation_matrix2d for center
#     angle = angle.expand(tensor.shape[0])
#     center = center.expand(tensor.shape[0], -1)
#     rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

#     # warp using the affine transform
#     return affine(tensor, rotation_matrix[..., :2, :3], mode, padding_mode, align_corners)



# 这个模块的主要目的是通过学习旋转等变换来提高模型性能，通常用于需要对输入图像进行适应性变换的任务。
# 在前向传播中，它首先将输入进行填充，然后旋转，最后反填充以生成输出。这有助于模型适应不同旋转条件下
# 的输入图像。模块中的旋转角度可以通过训练来优化，也可以通过train_angle参数控制是否训练。

# 这段代码定义了一个名为LearnableSpatialTransformWrapper的PyTorch模块，用于对输入图像进行学习的空间变换。
# 该模块可以包装一个实际的转换模块（impl），通常是一个卷积神经网络（ConvNet），以便学习对输入图像进行变换，
# 如旋转、平移等。这个模块还允许控制旋转角度、填充系数以及是否训练旋转角度。
class LearnableSpatialTransformWrapper(nn.Module):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    # impl：实际的变换模块，通常是一个卷积神经网络。
    # pad_coef：填充系数，用于指定填充的比例。
    # angle_init_range：旋转角度的初始范围。
    # train_angle：一个布尔值，指定是否训练旋转角度。

    def forward(self, x):
        # 前向传播函数，用于应用空间变换到输入数据。它可以处理不同类型的输入，如张量或元组，
        # 对不同类型的输入应用变换，并返回处理后的结果。
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        # 用于执行空间变换的函数。它首先进行填充，然后旋转输入图像。
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        # 用于执行反向的空间变换，以还原处理后的图像。它首先反向旋转，然后去除填充，以还原原始图像。
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y


if __name__ == '__main__':
    layer = LearnableSpatialTransformWrapper(nn.Identity())
    x = torch.arange(2* 3 * 15 * 15).view(2, 3, 15, 15).float()
    y = layer(x)
    assert x.shape == y.shape
    assert torch.allclose(x[:, :, 1:, 1:][:, :, :-1, :-1], y[:, :, 1:, 1:][:, :, :-1, :-1])
    print('all ok')
