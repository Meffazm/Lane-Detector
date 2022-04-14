import torch


def compute_hnet_loss(batch, coefs, order):
    losses = 0
    counter = 0
    idx = 0
    for pts in batch:
        for lane_pts in pts:
            if lane_pts.sum() != 0:
                loss = _compute_hnet_loss(lane_pts, coefs[idx], order=order)
                if loss:
                    losses += loss
                    counter += 1
        idx += 1
    return losses / counter if counter != 0 else 0


def _compute_hnet_loss(pts, coefs, order):
    """
        pts [2, M]: set of points in the original plane
        coefs [6]: contains coefficients of the H matrix
        order: polynomial order that used to fit the lane (2 or 3)
    """
    # adding 1 to the 3rd dim of each point and filtering zero values
    tresh = torch.nonzero(pts, as_tuple=True)[1].max().item()
    pts = pts[:, :tresh + 1]
    pts = torch.cat((pts, torch.ones(1, pts.size(1)).type_as(coefs)), 0)

    # convert 6 coefficients into H, which is a 3x3 matrix
    # H-matrix: [[a, b, c], [0, d, e], [0, f, 1]]

    H = torch.zeros(3, 3)
    H[0, 0] = coefs[0]
    H[0, 1] = coefs[1]
    H[0, 2] = coefs[2]
    H[1, 1] = coefs[3]
    H[1, 2] = coefs[4]
    H[2, 1] = coefs[5]
    H[2, 2] = 1
    H = H.type_as(coefs)

    # Transform all points from original plan to the mapped plane using H
    pts_transformed = torch.matmul(H, pts)

    # Scale output so that the 3rd is 1
    pts_transformed = torch.div(pts_transformed, pts_transformed[2:3, :])

    # Fitting lines in the transformed space using the close-formed solution
    X_p = pts_transformed[0, :]
    Y_p = pts_transformed[1, :]

    if order == 2:
        Y = torch.stack((Y_p * Y_p, Y_p,
                         torch.ones(pts.size(1)).type_as(coefs)), 1)  # Size: M x 3
    elif order == 3:
        Y = torch.stack((Y_p * Y_p * Y_p, Y_p * Y_p, Y_p,
                         torch.ones(pts.size(1)).type_as(coefs)), 1)  # Size: M x 3
    else:
        raise ValueError('Unknown order', order)
    try:
        W = torch.matmul(torch.matmul(torch.pinverse(torch.matmul(Y.transpose(0, 1), Y)),
                                      Y.transpose(0, 1)), X_p.unsqueeze(1))
    except:
        print(torch.matmul(Y.transpose(0, 1), Y))
        return

    # using the learned model to predict the x values in the projected plane
    X_pred = torch.matmul(Y, W)  # Size: M x 1
    # concat with Y and one's vector to form a tensor size: M x 3
    pts_pred = torch.cat((X_pred, Y_p.unsqueeze(1), torch.ones(pts.size(1), 1).type_as(coefs)), 1)
    # project it back to the original plane
    pts_back = torch.matmul(torch.inverse(H), pts_pred.transpose(0, 1))
    # Scale output so that the 3rd is 1
    pts_back = torch.div(pts_back, pts_back[2:3, :])

    # compute mse loss
    loss = torch.mean((pts[0, :] - pts_back[0, :]) ** 2)
    return loss
