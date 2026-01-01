import random
import os
import glob
import time
import warnings
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm.notebook import tqdm
from statistics import mean
import cv2

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR
from monai.metrics import compute_iou
from monai.metrics import compute_dice
from torch.cuda.amp import autocast, GradScaler

def train_test_model (optimizer : torch.optim.Optimizer,
                      loss_function : torch.nn.Module,
                      num_epochs : int,
                      train_dataloader : torch.utils.data.DataLoader,
                      test_dataloader : torch.utils.data.DataLoader,
                      model : torch.nn.Module,
                      device : str,
                      target_width : int,
                      target_height : int,
                      save=False,
                      main_model_title = "default",
                      save_every = 5) :

    model.to(device)
    model.train()

    print(f"Optimizer lr before scheduler initialization: {optimizer.param_groups[0]['lr']}")

    # # Initialize scheduler
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=0.01,
    #     epochs=num_epochs,
    #     steps_per_epoch=len(train_dataloader)
    # )

    # print(f"Optimizer lr after scheduler initialization : {optimizer.param_groups[0]['lr']}")


    # # Initialize scheduler
    # scheduler = CosineAnnealingLR (
    #     optimizer,
    #     T_max=num_epochs,
    #     eta_min=1e-7
    # )

    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min'
    )

    print(f"Optimizer lr after scheduler initialization : {optimizer.param_groups[0]['lr']}")

    # Lists to store metrics per epoch
    train_losses, train_ious, train_dices = [], [], []
    train_loss_std, train_iou_std, train_dice_std = [], [], []
    test_losses, test_ious, test_dices = [], [], []
    test_loss_std, test_iou_std, test_dice_std = [], [], []

    # Variable to keep track of lr value
    current_lr = 0
    lrs = []

    for epoch in range(num_epochs):

        print(f'------------------------------EPOCH: {epoch+1}------------------------------------')

        batch_losses = []
        batch_ious = []
        batch_dices = []

        #########################################
        ############## Train Loop ###############
        #########################################

        for i, batch in enumerate(tqdm(train_dataloader)):

            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)

            # process the output
            predicted_masks = outputs.pred_masks.squeeze(1)

            # adapt to proper mask dimensions

            predicted_masks = nn.functional.interpolate(predicted_masks,
                        size=(target_height, target_width),
                        mode='bilinear',
                        align_corners=False)
            
            predicted_masks.squeeze(1)

            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            # print(f"TRAIN UNIQUES : {np.unique(np.array(ground_truth_masks.cpu().numpy()))}")

            sam_masks_prob = torch.sigmoid(predicted_masks)
            sam_masks_prob = sam_masks_prob.squeeze()
            sam_masks = (sam_masks_prob > 0.5)

            # print(f"pred shape : {predicted_masks.shape}")
            # print(f"ground shape : {ground_truth_masks.shape}")

            # compute loss
            loss = loss_function(predicted_masks, ground_truth_masks.unsqueeze(1))
            batch_losses.append(loss.item())

            sam_masks = sam_masks.unsqueeze(1)

            if not(sam_masks.shape == (ground_truth_masks.unsqueeze(1)).shape) :
                sam_masks = sam_masks.permute(1, 0, 2).unsqueeze(0)


            ious = compute_iou(sam_masks,
                                ground_truth_masks.unsqueeze(1), ignore_empty=False)
            
            dices = compute_dice(
                sam_masks, ground_truth_masks.unsqueeze(1), ignore_empty=False
            )

            # Convert to float and append, using nanmean to avoid NaNs
            batch_ious.append(torch.nanmean(ious).cpu().item())
            batch_dices.append(torch.nanmean(dices).cpu().item())

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute epoch metrics: mean + std
        train_losses.append(np.nanmean(batch_losses))
        train_loss_std.append(np.nanstd(batch_losses))
        train_ious.append(np.nanmean(batch_ious))
        train_iou_std.append(np.nanstd(batch_ious))
        train_dices.append(np.nanmean(batch_dices))
        train_dice_std.append(np.nanstd(batch_dices))

        # Get Current lr value to track its updates
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)

        print(f"Learning Rate: {current_lr}")
        print(f"Train Loss: {train_losses[-1]:.4f} ± {train_loss_std[-1]:.4f}")
        print(f"Train IoU: {train_ious[-1]:.4f} ± {train_iou_std[-1]:.4f}")
        print(f"Train Dice: {train_dices[-1]:.4f} ± {train_dice_std[-1]:.4f}")

        #########################################
        ############## Test Loop ################
        #########################################
        batch_losses = []
        batch_ious = []
        batch_dices = []

        model.eval()

        # Iteratire through test images
        with torch.no_grad():

            for batch in tqdm(test_dataloader):

                # forward pass
                outputs = model(pixel_values=batch["pixel_values"].cuda(),
                                input_boxes=batch["input_boxes"].cuda(),
                                multimask_output=False)

                predicted_masks = outputs.pred_masks.squeeze(1)

                # adapt to proper mask dimensions
                predicted_masks = nn.functional.interpolate(predicted_masks,
                        size=(target_height, target_width),
                        mode='bilinear',
                        align_corners=False)

                ground_truth_masks = batch["ground_truth_mask"].float().cuda()


                # apply sigmoid
                sam_mask_prob = torch.sigmoid(predicted_masks)
                sam_mask_prob = sam_mask_prob.cpu().numpy().squeeze()
                sam_mask = (sam_mask_prob > 0.5).astype(np.uint8)

                sam_mask = torch.tensor(sam_mask, device = device).unsqueeze(0).unsqueeze(0)

                loss = loss_function(predicted_masks, ground_truth_masks.unsqueeze(1))
                batch_losses.append(loss.cpu().item())

                iou = compute_iou(sam_mask,
                                    ground_truth_masks.unsqueeze(1))
                
                dice = compute_dice (
                    sam_mask, ground_truth_masks.unsqueeze(1)
                )


                sam_mask = sam_mask.squeeze(0).squeeze(0)
                batch_ious.append(torch.nanmean(iou).cpu().item())
                batch_dices.append(torch.nanmean(dice).cpu().item())

                # #Step the scheduler using val loss
                # scheduler.step(loss.cpu().item())  # reduce LR when val loss stops improving

        # Compute test epoch metrics
        test_losses.append(np.nanmean(batch_losses))
        test_loss_std.append(np.nanstd(batch_losses))
        test_ious.append(np.nanmean(batch_ious))
        test_iou_std.append(np.nanstd(batch_ious))
        test_dices.append(np.nanmean(batch_dices))
        test_dice_std.append(np.nanstd(batch_dices))

        # scheduler.step(test_losses[-1])

        print(f"Test Loss: {test_losses[-1]:.4f} ± {test_loss_std[-1]:.4f}")
        print(f"Test IoU: {test_ious[-1]:.4f} ± {test_iou_std[-1]:.4f}")
        print(f"Test Dice: {test_dices[-1]:.4f} ± {test_dice_std[-1]:.4f}")

        #########################################
        ############## MODEL SAVING #############
        #########################################

        if (save) :
            if ((epoch+1)%save_every == 0) :
                # Define the folder and ensure it exists
                save_dir = f'models/models_{main_model_title}'
                os.makedirs(save_dir, exist_ok=True)

                # Define the checkpoint path
                checkpoint_path = os.path.join(
                    save_dir, f'focalloss_lr1e-4_sched-ReduceLr_stratify{epoch+1}.pth'
                )

                # Save the model parameters
                torch.save(model.state_dict(), checkpoint_path)

                print("----------------------------------------------------")
                print("------------------- Model Saved! -------------------")
                print("----------------------------------------------------\n")

        model.train()

    #########################################
    ######## SAVE METRICS TO CSV ############
    #########################################
    save_dir = "Losses, IoUs and Dices"
    os.makedirs(save_dir, exist_ok=True)

    train_losses = [float(x) for x in train_losses]
    train_ious = [float(x) for x in train_ious]
    train_dices = [float(x) for x in train_dices]
    test_losses = [float(x) for x in test_losses]
    test_ious = [float(x) for x in test_ious]
    test_dices = [float(x) for x in test_dices]

    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Train_Loss": train_losses,
        "Train_Loss_STD": train_loss_std,
        "Train_IoU": train_ious,
        "Train_IoU_STD": train_iou_std,
        "Train_Dice": train_dices,
        "Train_Dice_STD": train_dice_std,
        "Test_Loss": test_losses,
        "Test_Loss_STD": test_loss_std,
        "Test_IoU": test_ious,
        "Test_IoU_STD": test_iou_std,
        "Test_Dice": test_dices,
        "Test_Dice_STD": test_dice_std
    })

    metrics_path = os.path.join(save_dir, f"training_metrics_{main_model_title}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"📊 Metrics saved to: {metrics_path}")

    # #########################################
    # ######## BACKUP MODEL SAVING ############
    # #########################################

    # checkpoint_path = f'models/sam_huge_backup_{num_epochs}.pth'

    # # Save the parameters of the entire model
    # torch.save(model.state_dict(), checkpoint_path)
    # print("----------------------------------------------------")
    # print("------------------- Model Saved! -------------------")
    # print("----------------------------------------------------")

            
    return train_losses, train_ious, train_dices,\
                test_losses, test_ious, test_dices,\
                    train_loss_std, train_iou_std, train_dice_std,\
                        test_loss_std, test_iou_std, test_dice_std, lrs

def test_with_visualization(
    test_dataloader: torch.utils.data.DataLoader,
    ds: pd.DataFrame,
    model: torch.nn.Module,
    device: str,
    target_width: int,
    target_height: int,
    save: bool = False,
    visualize: bool = True,
    morph: bool = True,
    save_dir: str = "preds"
):
    import os
    import cv2
    import torch
    import numpy as np
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    test_ious = []
    test_dices = []
    iou_results = {}
    dice_results = {}
    pred_paths = {}

    model.eval()

    # ============================
    # Iterate through test images
    # ============================
    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            outputs = model(
                pixel_values=batch["pixel_values"].cuda(),
                input_boxes=batch["input_boxes"].cuda(),
                multimask_output=False
            )

            ground_truth_masks = batch["ground_truth_mask"].float().cuda()

            # Sigmoid + threshold
            sam_mask_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            sam_mask_prob = sam_mask_prob.cpu().numpy().squeeze()
            sam_mask = (sam_mask_prob > 0.5).astype(np.uint8)

            sam_mask = torch.tensor(sam_mask, device=device).unsqueeze(0).unsqueeze(0)

            sam_mask = nn.functional.interpolate(
                sam_mask,
                size=(target_height, target_width),
                mode="nearest"
            )

            iou = compute_iou(sam_mask, ground_truth_masks.unsqueeze(1))
            dice = compute_dice(sam_mask, ground_truth_masks.unsqueeze(1))

            sam_mask = sam_mask.squeeze(0).squeeze(0)

            # ============================
            # Morphological post-processing
            # ============================
            if morph:
                sam_mask_np = sam_mask.detach().cpu().numpy().astype(np.uint8)

                sam_mask_np = cv2.morphologyEx(
                    sam_mask_np,
                    cv2.MORPH_CLOSE,
                    np.ones((5, 5), np.uint8)
                )

                sam_mask_np = cv2.morphologyEx(
                    sam_mask_np,
                    cv2.MORPH_OPEN,
                    np.ones((3, 3), np.uint8)
                )

                sam_mask = torch.tensor(sam_mask_np, device=device)

            test_ious.append(iou)
            test_dices.append(dice)

            # ============================
            # Match image → mask in dataset
            # ============================
            scan_filename = os.path.basename(batch["filename"][0])
            mask_row = ds[ds["image_path"].str.endswith(scan_filename)]

            if len(mask_row) == 1:
                mask_filename = os.path.basename(mask_row["mask_path"].values[0])
                iou_results[mask_filename] = float(iou.squeeze())
                dice_results[mask_filename] = float(dice.squeeze())

            # ============================
            # SAVE PREDICTED MASK
            # ============================
            if save:
                base_save_dir = save_dir  # 🔑 ROOT NEVER CHANGES

                mask_path = batch["filename"][0]
                parts = os.path.normpath(mask_path).split(os.sep)

                try:
                    subject = parts[-4]
                    week = parts[-3]
                except IndexError:
                    subject, week = "Unknown", "Unknown"

                mask_filename = os.path.basename(mask_path)
                pred_name = (
                    mask_filename
                    .replace("_w", "")
                    .replace(".png", "_mask_pred.png")
                )

                batch_save_dir = os.path.join(
                    base_save_dir,
                    str(subject),
                    str(week),
                    "Predictions"
                )
                os.makedirs(batch_save_dir, exist_ok=True)

                save_path = os.path.join(batch_save_dir, pred_name)

                sam_mask_np = sam_mask.detach().cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(save_path, sam_mask_np)

                if len(mask_row) == 1:
                    pred_paths[mask_filename] = save_path

            # ============================
            # Visualization
            # ============================
            if visualize:
                print(f"IoU: {float(iou)}")
                print(f"Dice: {float(dice)}")

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(batch["pixel_values"][0, 1], cmap="gray")
                plt.title("MRI Scan")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(batch["ground_truth_mask"][0], cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(sam_mask.cpu(), cmap="gray")
                plt.title("Prediction")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

    # ============================
    # Add results to dataframe
    # ============================
    ds = ds.copy()

    ds["IoU"] = ds["mask_path"].apply(
        lambda p: iou_results.get(os.path.basename(p), np.nan)
    )

    ds["Dice"] = ds["mask_path"].apply(
        lambda p: dice_results.get(os.path.basename(p), np.nan)
    )

    ds["pred_mask_path"] = ds["mask_path"].apply(
        lambda p: pred_paths.get(os.path.basename(p), np.nan)
    )

    print(
        f"Average IoU over test set: "
        f"{np.nanmean([t.cpu().item() if torch.is_tensor(t) else t for t in test_ious])}"
    )

    print(
        f"Average Dice over test set: "
        f"{np.nanmean([t.cpu().item() if torch.is_tensor(t) else t for t in test_dices])}"
    )

    return ds


def patch_set_image_for_medsam2(predictor):
    """
    Patches SAM2ImagePredictor.set_image to:
    1. Accept PyTorch tensors or NumPy arrays.
    2. Resize to 512x512.
    3. Ensure HWC float32, 0-255 range.
    4. Fix stride errors with reshape fallback.
    """
    original_set_image = predictor.set_image

    def patched_set_image(image):
        # --- Step 1: Convert tensor -> NumPy HWC ---
        if isinstance(image, np.ndarray):
            img = image
        elif isinstance(image, torch.Tensor):
            img = image.permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unsupported image type {type(image)}")

        # --- Step 2: Resize to 512x512 ---
        img = cv2.resize(img, (512, 512))

        # --- Step 3: Ensure float32 HWC, 0-255 range ---
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img *= 255.0  # scale normalized image to 0-255

        # --- Step 4: Call original set_image (NumPy only) ---
        try:
            return original_set_image(img)
        except RuntimeError as e:
            # fallback for stride/view errors
            if "view size is not compatible" in str(e):
                feats = predictor._features["high_res_feats"]
                predictor._features["high_res_feats"] = [f.reshape(1, -1, *f.shape[1:]) for f in feats]
                predictor._features["image_embed"] = [predictor._features["image_embed"][-1].reshape(
                    1, -1, *predictor._features["image_embed"][-1].shape[1:])]
                return predictor._features
            else:
                raise e

    predictor.set_image = patched_set_image


# def forward_sam2(predictor, image, box, device, is_medsam2):
#     """
#     Performs forward pass for SAM2 predictor given a single image and bounding box.
#     Returns predicted masks and predicted scores.
#     """

#     # example: single foreground point at center
#     input_point = np.array([[256, 256]])  # HWC coordinates, after your 512x512 resize
#     input_label = np.array([1]) 

#     # ✅ Fix for MedSAM2: ensure contiguous tensor before set_image
#     if is_medsam2 :
#         print("HERE")
#         patch_set_image_for_medsam2(predictor)

#     predictor.set_image(image)

#      # Prepare prompts (box-based)
#     mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
#         input_point, input_label, box=box[None, :], mask_logits=None, normalize_coords=True
#     )

#     # If no valid prompts were generated, skip
#     if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
#         return None, None

#     sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder (
#         points=(unnorm_coords, labels),
#         boxes=None,
#         masks=None
#     )

#     high_res_feats = [lvl[-1].unsqueeze(0) for lvl in predictor._features["high_res_feats"]]
#     low_res_masks, pred_scores, _, _ = predictor.model.sam_mask_decoder(
#         image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
#         image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
#         sparse_prompt_embeddings=sparse_embeddings,
#         dense_prompt_embeddings=dense_embeddings,
#         multimask_output=True,
#         repeat_image=False,
#         high_res_features=high_res_feats,
#     )

#     prd_masks = predictor._transforms.postprocess_masks(
#         low_res_masks, predictor._orig_hw[-1]
#     )

#     return prd_masks.to(device), pred_scores.to(device)

def forward_sam2(predictor, image, box, device, is_medsam2):
    """
    Forward pass for SAM2 predictor.
    Converts input to a format supported by SAM2ImagePredictor.
    """
    # Convert torch.Tensor -> numpy HWC
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.ndim == 3:  # C,H,W -> H,W,C
            if image.shape[0] in [1, 3]:  # channels first
                image = np.transpose(image, (1, 2, 0))
            else:  # probably already H,W,C
                pass
        image = image.astype(np.float32)
        # normalize if original tensor was uint8
        if image.max() > 1.0:
            image /= 255.0

    # Ensure HWC shape with 1 or 3 channels
    if image.ndim != 3 or image.shape[2] not in [1, 3]:
        raise ValueError(f"Image must be HWC with 1 or 3 channels, got shape {image.shape}")

    # MedSAM2 patch
    if is_medsam2:
        patch_set_image_for_medsam2(predictor)

    # Set image
    predictor.set_image(image)

    # Example single foreground point (required by SAM2 prompts)
    input_point = np.array([[256, 256]])
    input_label = np.array([1])

    # Prepare prompts (box-based)
    mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
        input_point, input_label, box=box[None, :], mask_logits=None, normalize_coords=True
    )

    if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
        return None, None

    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels),
        boxes=None,
        masks=None
    )

    high_res_feats = [lvl[-1].unsqueeze(0) for lvl in predictor._features["high_res_feats"]]
    low_res_masks, pred_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=False,
        high_res_features=high_res_feats,
    )

    prd_masks = predictor._transforms.postprocess_masks(
        low_res_masks, predictor._orig_hw[-1]
    )

    return prd_masks.to(device), pred_scores.to(device)



def train_test_sam2(optimizer: torch.optim.Optimizer,
                    loss_function : torch.nn.Module,
                    num_epochs: int,
                    train_dataloader: torch.utils.data.DataLoader,
                    test_dataloader: torch.utils.data.DataLoader,
                    predictor,  # SAM2ImagePredictor
                    device: str,
                    target_width: int,
                    target_height: int,
                    save=False,
                    is_medsam2 = False,
                    save_every=5):
    
    print("#########################################")
    print(f"######## TRAINING FOR {num_epochs} EPOCHS #########")
    print("#########################################")
    print("\n")

    predictor.model.to(device)
    predictor.model.train()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
    scaler = GradScaler()

    # Metric containers
    train_losses, train_ious, train_dices = [], [], []
    train_loss_std, train_iou_std, train_dice_std = [], [], []
    test_losses, test_ious, test_dices = [], [], []
    test_loss_std, test_iou_std, test_dice_std = [], [], []

    # ===================== TRAINING LOOP =====================
    for epoch in range(num_epochs):
        print(f'------------------------------EPOCH: {epoch+1}------------------------------------')

        predictor.model.train()
        batch_losses, batch_ious, batch_dices = [], [], []

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad(set_to_none=True)

            # batch["pixel_values"] = batch["pixel_values"].permute(0, 3, 1, 2).float()

            # with autocast():
            loss_accum, iou_accum = 0.0, 0.0

            # Loop through each item in batch
            for b in range(len(batch["pixel_values"])):
                image = batch["pixel_values"][b].to(device)  # already CxHxW
                # image = batch["pixel_values"][b].cpu().numpy()
                # print(f"image shape : {image.shape}")
                gt_mask = batch["ground_truth_mask"][b].to(device).float()
                # print(f"gt shape : {gt_mask.shape}")
                box = batch["input_boxes"][b].to(device)
                # print(f"box shape : {box.shape}")

                pred_masks, pred_scores = forward_sam2(predictor, image, box, device, is_medsam2)
                prd_mask = torch.sigmoid(pred_masks[:, 0]).squeeze(0)

                seg_loss = loss_function(prd_mask, gt_mask)
                iou_val = compute_iou(prd_mask, gt_mask)
                score_loss = torch.abs(pred_scores[:, 0] - iou_val).mean()

                total_loss = seg_loss + 0.05 * score_loss
                loss_accum += total_loss
                iou_accum += iou_val

            total_loss = loss_accum / len(batch["pixel_values"])
            avg_iou = iou_accum / len(batch["pixel_values"])

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # batch_ious.append(iou.detach().cpu().item())
            # batch_dices.append(dice.detach().cpu().item())
            # batch_losses.append(loss.detach().cpu().item())

            batch_losses.append(total_loss.item())
            batch_ious.append(avg_iou.detach().cpu().item())
            batch_dices.append(avg_iou.detach().cpu().item())  # Dice ≈ IoU for single-class

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_losses.append(np.nanmean(batch_losses))
        train_loss_std.append(np.nanstd(batch_losses))
        train_ious.append(np.nanmean(batch_ious))
        train_iou_std.append(np.nanstd(batch_ious))
        train_dices.append(np.nanmean(batch_dices))
        train_dice_std.append(np.nanstd(batch_dices))

        print(f"Train Loss: {train_losses[-1]:.4f} ± {train_loss_std[-1]:.4f}")
        print(f"Train IoU: {train_ious[-1]:.4f} ± {train_iou_std[-1]:.4f}")
        print(f"Train Dice: {train_dices[-1]:.4f} ± {train_dice_std[-1]:.4f}")

        # ===================== TEST LOOP =====================
        predictor.model.eval()
        batch_losses, batch_ious, batch_dices = [], [], []

        # with torch.no_grad(), autocast():
        for batch in tqdm(test_dataloader):
            loss_accum, iou_accum = 0.0, 0.0
            batch["pixel_values"] = batch["pixel_values"].permute(0, 3, 1, 2).float()

            for b in range(len(batch["pixel_values"])):
                image = batch["pixel_values"][b].cpu().numpy().transpose(1, 2, 0)
                gt_mask = batch["ground_truth_mask"][b].cpu().numpy()
                box = batch["input_boxes"][b].cpu().numpy()

                pred_masks, pred_scores = forward_sam2(predictor, image, box, device, is_medsam2)
                prd_mask = torch.sigmoid(pred_masks[:, 0]).squeeze(0)
                gt_mask_t = torch.tensor(gt_mask.astype(np.float32)).to(device)

                seg_loss = loss_function(prd_mask, gt_mask_t)
                iou_val = compute_iou(prd_mask, gt_mask_t)
                score_loss = torch.abs(pred_scores[:, 0] - iou_val).mean()

                total_loss = seg_loss + 0.05 * score_loss
                loss_accum += total_loss
                iou_accum += iou_val

            total_loss = loss_accum / len(batch["pixel_values"])
            avg_iou = iou_accum / len(batch["pixel_values"])

            batch_losses.append(total_loss.item())
            batch_ious.append(avg_iou.detach().cpu().item())
            batch_dices.append(avg_iou.detach().cpu().item())  # Dice ≈ IoU for single-class

        test_losses.append(np.nanmean(batch_losses))
        test_loss_std.append(np.nanstd(batch_losses))
        test_ious.append(np.nanmean(batch_ious))
        test_iou_std.append(np.nanstd(batch_ious))
        test_dices.append(np.nanmean(batch_dices))
        test_dice_std.append(np.nanstd(batch_dices))

        print(f"Test Loss: {test_losses[-1]:.4f} ± {test_loss_std[-1]:.4f}")
        print(f"Test IoU: {test_ious[-1]:.4f} ± {test_iou_std[-1]:.4f}")
        print(f"Test Dice: {test_dices[-1]:.4f} ± {test_dice_std[-1]:.4f}")

        scheduler.step(test_losses[-1])

        # ===================== SAVE MODEL =====================
        if save and ((epoch + 1) % save_every == 0):
            os.makedirs("models", exist_ok=True)
            checkpoint_path = f"models/sam2_{epoch+1}.pth"
            torch.save(predictor.model.state_dict(), checkpoint_path)
            print("----------------------------------------------------")
            print("------------------- Model Saved! -------------------")
            print("----------------------------------------------------\n")

        predictor.model.train()

    # ===================== SAVE METRICS =====================
    save_dir = "Losses, IoUs and Dices"
    os.makedirs(save_dir, exist_ok=True)

    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Train_Loss": train_losses,
        "Train_Loss_STD": train_loss_std,
        "Train_IoU": train_ious,
        "Train_IoU_STD": train_iou_std,
        "Train_Dice": train_dices,
        "Train_Dice_STD": train_dice_std,
        "Test_Loss": test_losses,
        "Test_Loss_STD": test_loss_std,
        "Test_IoU": test_ious,
        "Test_IoU_STD": test_iou_std,
        "Test_Dice": test_dices,
        "Test_Dice_STD": test_dice_std
    })

    metrics_path = os.path.join(save_dir, "training_metrics_sam2.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"📊 Metrics saved to: {metrics_path}")

    return train_losses, train_ious, train_dices, \
           test_losses, test_ious, test_dices, \
           train_loss_std, train_iou_std, train_dice_std, \
           test_loss_std, test_iou_std, test_dice_std

def train_test_sam2_v2(optimizer: torch.optim.Optimizer,
                        loss_function: torch.nn.Module,
                        num_epochs: int,
                        train_dataloader: torch.utils.data.DataLoader,
                        test_dataloader: torch.utils.data.DataLoader,
                        predictor,  # SAM2ImagePredictor
                        device: str,
                        target_width: int,
                        target_height: int,
                        save=False,
                        is_medsam2=False,
                        save_every=5):

    print(f"######## TRAINING FOR {num_epochs} EPOCHS #########\n")

    predictor.model.to(device)
    predictor.model.train()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    scaler = torch.cuda.amp.GradScaler()

    # Metrics
    train_losses, train_ious, train_dices = [], [], []
    train_loss_std, train_iou_std, train_dice_std = [], [], []
    test_losses, test_ious, test_dices = [], [], []
    test_loss_std, test_iou_std, test_dice_std = [], [], []

    # ===================== TRAIN LOOP =====================
    for epoch in range(num_epochs):
        print(f'------------------ EPOCH: {epoch+1} ------------------')
        predictor.model.train()
        batch_losses, batch_ious, batch_dices = [], [], []

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad(set_to_none=True)

            loss_accum, iou_accum = 0.0, 0.0

            for b in range(len(batch["pixel_values"])):
                image = batch["pixel_values"][b].float()
                image = image.to(device)
                gt_mask = batch["ground_truth_mask"][b].to(device).float()
                box = batch["input_boxes"][b].to(device)

                pred_masks, pred_scores = forward_sam2(predictor, image, box, device, is_medsam2)
                if pred_masks is None:
                    continue

                prd_mask = torch.sigmoid(pred_masks[:, 0]).squeeze(0)

                seg_loss = loss_function(prd_mask, gt_mask)
                iou_val = compute_iou(prd_mask, gt_mask)
                score_loss = torch.abs(pred_scores[:, 0] - iou_val).mean()

                total_loss = seg_loss + 0.05 * score_loss
                loss_accum += total_loss
                iou_accum += iou_val

            total_loss = loss_accum / len(batch["pixel_values"])
            avg_iou = iou_accum / len(batch["pixel_values"])

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_losses.append(total_loss.item())
            batch_ious.append(avg_iou.detach().cpu().item())
            batch_dices.append(avg_iou.detach().cpu().item())  # Dice ≈ IoU for single-class

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save train metrics
        train_losses.append(np.nanmean(batch_losses))
        train_loss_std.append(np.nanstd(batch_losses))
        train_ious.append(np.nanmean(batch_ious))
        train_iou_std.append(np.nanstd(batch_ious))
        train_dices.append(np.nanmean(batch_dices))
        train_dice_std.append(np.nanstd(batch_dices))

        print(f"Train Loss: {train_losses[-1]:.4f} ± {train_loss_std[-1]:.4f}")
        print(f"Train IoU: {train_ious[-1]:.4f} ± {train_iou_std[-1]:.4f}")
        print(f"Train Dice: {train_dices[-1]:.4f} ± {train_dice_std[-1]:.4f}")

        # ===================== TEST LOOP =====================
        predictor.model.eval()
        batch_losses, batch_ious, batch_dices = [], [], []

        for batch in tqdm(test_dataloader):
            loss_accum, iou_accum = 0.0, 0.0

            for b in range(len(batch["pixel_values"])):
                image = batch["pixel_values"][b].float() / 255.0  # ✅ Convert to float32
                image = image.to(device)
                gt_mask = batch["ground_truth_mask"][b].to(device).float()
                box = batch["input_boxes"][b].to(device)

                with torch.no_grad():
                    pred_masks, pred_scores = forward_sam2(predictor, image, box, device, is_medsam2)
                    if pred_masks is None:
                        continue

                    prd_mask = torch.sigmoid(pred_masks[:, 0]).squeeze(0)

                    seg_loss = loss_function(prd_mask, gt_mask)
                    iou_val = compute_iou(prd_mask, gt_mask)
                    score_loss = torch.abs(pred_scores[:, 0] - iou_val).mean()

                    total_loss = seg_loss + 0.05 * score_loss
                    loss_accum += total_loss
                    iou_accum += iou_val

            total_loss = loss_accum / len(batch["pixel_values"])
            avg_iou = iou_accum / len(batch["pixel_values"])

            batch_losses.append(total_loss.item())
            batch_ious.append(avg_iou.detach().cpu().item())
            batch_dices.append(avg_iou.detach().cpu().item())

        # Save test metrics
        test_losses.append(np.nanmean(batch_losses))
        test_loss_std.append(np.nanstd(batch_losses))
        test_ious.append(np.nanmean(batch_ious))
        test_iou_std.append(np.nanstd(batch_ious))
        test_dices.append(np.nanmean(batch_dices))
        test_dice_std.append(np.nanstd(batch_dices))

        print(f"Test Loss: {test_losses[-1]:.4f} ± {test_loss_std[-1]:.4f}")
        print(f"Test IoU: {test_ious[-1]:.4f} ± {test_iou_std[-1]:.4f}")
        print(f"Test Dice: {test_dices[-1]:.4f} ± {test_dice_std[-1]:.4f}")

        scheduler.step(test_losses[-1])

        # Save model
        if save and ((epoch + 1) % save_every == 0):
            os.makedirs("models", exist_ok=True)
            checkpoint_path = f"models/sam2_{epoch+1}.pth"
            torch.save(predictor.model.state_dict(), checkpoint_path)
            print("-------------- Model Saved! ------------------")

    # Save metrics CSV
    save_dir = "Losses, IoUs and Dices"
    os.makedirs(save_dir, exist_ok=True)
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Train_Loss": train_losses,
        "Train_Loss_STD": train_loss_std,
        "Train_IoU": train_ious,
        "Train_IoU_STD": train_iou_std,
        "Train_Dice": train_dices,
        "Train_Dice_STD": train_dice_std,
        "Test_Loss": test_losses,
        "Test_Loss_STD": test_loss_std,
        "Test_IoU": test_ious,
        "Test_IoU_STD": test_iou_std,
        "Test_Dice": test_dices,
        "Test_Dice_STD": test_dice_std
    })
    metrics_path = os.path.join(save_dir, "training_metrics_sam2.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"📊 Metrics saved to: {metrics_path}")

    return (train_losses, train_ious, train_dices,
            test_losses, test_ious, test_dices,
            train_loss_std, train_iou_std, train_dice_std,
            test_loss_std, test_iou_std, test_dice_std)

def plot_train_validation_curves(
    train_vals : list,
    validation_vals : list,
    type : str,
    train_vals_std=None,
    validation_vals_std=None,
    show_std=False):
    """
    Plots training and test losses over epochs, optionally with standard deviation error bars.

    Parameters
    ----------
    train_losses : list or np.ndarray
        Mean training loss per epoch.
    test_losses : list or np.ndarray
        Mean test loss per epoch.
    train_loss_std : list or np.ndarray, optional
        Standard deviation of training loss per epoch.
    test_loss_std : list or np.ndarray, optional
        Standard deviation of test loss per epoch.
    show_std : bool, default=True
        Whether to plot standard deviation as error bars.
    """

    epochs = np.arange(1, len(train_vals) + 1)
    plt.figure(figsize=(8, 6))

    if show_std and train_vals_std is not None and validation_vals_std is not None:
        # Plot with error bars
        plt.errorbar(
            epochs, train_vals, yerr=train_vals_std,
            label=f'Train {type}', linewidth=2, capsize=4, fmt='-o'
        )
        plt.errorbar(
            epochs, validation_vals, yerr=validation_vals_std,
            label=f'Validation {type}', linewidth=2, linestyle='--', capsize=4, fmt='-s'
        )
    else:
        # Plot without error bars
        plt.plot(epochs, train_vals, '-o', label=f'Train {type}', linewidth=2)
        plt.plot(epochs, validation_vals, '-s', label=f'Validation {type}', linewidth=2)

    plt.title(f'Training and Validation {type} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.tight_layout()
    plt.show()

    epochs = np.arange(1, len(train_vals) + 1)
    plt.figure(figsize=(8, 6))

    if show_std and train_vals_std is not None and validation_vals_std is not None:
        # Plot with error bars
        plt.errorbar(
            epochs, train_vals, yerr=train_vals_std,
            label=f'Train {type}', linewidth=2, capsize=4, fmt='-o'
        )
        plt.errorbar(
            epochs, validation_vals, yerr=validation_vals_std,
            label=f'Validation {type}', linewidth=2, linestyle='--', capsize=4, fmt='-s'
        )
    else:
        # Plot without error bars
        plt.plot(epochs, train_vals, '-o', label=f'Train {type}', linewidth=2)
        plt.plot(epochs, validation_vals, '-s', label=f'Validation {type}', linewidth=2)

    plt.title(f'Training and Validation {type} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_train_validation_curves_NEW(
    train_vals: list,
    validation_vals: list,
    type: str,
    train_vals_std=None,
    validation_vals_std=None,
    show_std=False,
    save_path: str = None  # <-- new optional argument
):
    import numpy as np
    import matplotlib.pyplot as plt

    train_vals = np.array(train_vals)
    validation_vals = np.array(validation_vals)

    epochs = np.arange(1, len(train_vals) + 1)
    plt.figure(figsize=(8, 6))

    if show_std and train_vals_std is not None and validation_vals_std is not None:
        train_vals_std = np.array(train_vals_std)
        validation_vals_std = np.array(validation_vals_std)

        plt.errorbar(
            epochs, train_vals, yerr=train_vals_std,
            label=f'Train {type}', linewidth=2, capsize=4, fmt='-o'
        )
        plt.errorbar(
            epochs, validation_vals, yerr=validation_vals_std,
            label=f'Validation {type}', linewidth=2, linestyle='--', capsize=4, fmt='-s'
        )
    else:
        plt.plot(epochs, train_vals, '-o', label=f'Train {type}', linewidth=2)
        plt.plot(epochs, validation_vals, '-s', label=f'Validation {type}', linewidth=2)

    plt.title(f'Training and Validation {type} over Epochs')
    plt.xlabel('Epoch')

    # --------------------------------------------------
    # Metric-aware y-axis handling
    # --------------------------------------------------
    if type.lower() in ["iou", "dice"]:
        assert np.max(train_vals) <= 1.0 and np.max(validation_vals) <= 1.0, \
            f"{type} values must be in [0, 1]"

        if show_std and train_vals_std is not None and validation_vals_std is not None:
            ymin = min(
                np.min(train_vals - train_vals_std),
                np.min(validation_vals - validation_vals_std)
            )
        else:
            ymin = min(np.min(train_vals), np.min(validation_vals))

        margin = 0.05
        ymin = max(0.0, ymin - margin)

        plt.ylabel(f'{type} (%)')
        plt.ylim(ymin, 1.0)
        yticks = np.linspace(ymin, 1.0, 6)
        plt.yticks([y for y in yticks], [f"{int(y * 100)}%" for y in yticks])
    else:
        plt.ylabel(type)

    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.tight_layout()

    # -----------------------------
    # Save figure if path provided
    # -----------------------------
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()



def plot_lrs(
    train_vals : list,
    validation_vals : list,
    type : str,
    train_vals_std=None,
    validation_vals_std=None,
    show_std=False):
    """
    Plots training and test losses over epochs, optionally with standard deviation error bars.

    Parameters
    ----------
    train_losses : list or np.ndarray
        Mean training loss per epoch.
    test_losses : list or np.ndarray
        Mean test loss per epoch.
    train_loss_std : list or np.ndarray, optional
        Standard deviation of training loss per epoch.
    test_loss_std : list or np.ndarray, optional
        Standard deviation of test loss per epoch.
    show_std : bool, default=True
        Whether to plot standard deviation as error bars.
    """
    epochs = np.arange(1, len(train_vals) + 1)
    plt.figure(figsize=(8, 6))

    if show_std and train_vals_std is not None and validation_vals_std is not None:
        # Plot with error bars
        plt.errorbar(
            epochs, train_vals, yerr=train_vals_std,
            label=f'Train {type}', linewidth=2, capsize=4, fmt='-o'
        )
        plt.errorbar(
            epochs, validation_vals, yerr=validation_vals_std,
            label=f'Validation {type}', linewidth=2, linestyle='--', capsize=4, fmt='-s'
        )
    else:
        # Plot without error bars
        plt.plot(epochs, train_vals, '-o', label=f'Train {type}', linewidth=2)
        plt.plot(epochs, validation_vals, '--s', label=f'Validation {type}', linewidth=2)

    plt.title(f'Training and Validation {type} over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_lr_schedule(lr_values, title="Learning Rate Schedule", per_epoch=True):
    """
    Plot the learning rate schedule over epochs or steps.

    Parameters
    ----------
    lr_values : list or np.ndarray
        List of learning rate values recorded during training.
    title : str, optional
        Title for the plot.
    per_epoch : bool, optional
        If True, x-axis is in epochs. If False, it's in steps.
    """
    plt.figure(figsize=(8, 5))
    x = np.arange(1, len(lr_values) + 1)
    plt.plot(x, lr_values, linewidth=2, color='royalblue')
    plt.xlabel("Epoch" if per_epoch else "Step")
    plt.ylabel("Learning Rate")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def train_test_fusion(optimizer : torch.optim.Optimizer,
                      loss_function : torch.nn.Module,
                      num_epochs : int,
                      train_dataloader : torch.utils.data.DataLoader,
                      test_dataloader : torch.utils.data.DataLoader,
                      model : torch.nn.Module,
                      device : str,
                      target_width : int,
                      target_height : int,
                      save=False,
                      main_model_title="fusion_model",
                      save_every=5
):

    model.to(device)
    model.train()

    print(f"Optimizer lr before scheduler initialization: {optimizer.param_groups[0]['lr']}")

    # scheduler = OneCycleLR(...)
    # scheduler = CosineAnnealingLR(...)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min'
    )

    print(f"Optimizer lr after scheduler initialization : {optimizer.param_groups[0]['lr']}")

    # metrics
    train_losses, train_ious, train_dices = [], [], []
    train_loss_std, train_iou_std, train_dice_std = [], [], []
    test_losses, test_ious, test_dices = [], [], []
    test_loss_std, test_iou_std, test_dice_std = [], [], []

    lrs = []

    # ================================================================ #
    #                           EPOCH LOOP                             #
    # ================================================================ #
    for epoch in range(num_epochs):
        print(f"\n--------------------- EPOCH {epoch+1}/{num_epochs} ---------------------")
        batch_losses, batch_ious, batch_dices = [], [], []

        model.train()

        # ================================================================ #
        #                          TRAIN LOOP                              #
        # ================================================================ #
        for batch in tqdm(train_dataloader, desc="Training"):

            # ------------------------------------------------------------ #
            #                      FWD PASS (FUSION)                      #
            # ------------------------------------------------------------ #

            outputs = model(
                batch["fat"]["pixel_values"].to(device),
                batch["water"]["pixel_values"].to(device),
                batch["water"]["input_boxes"].to(device)
            )
            predicted_masks = outputs[0]

            predicted_masks = nn.functional.interpolate(
                predicted_masks,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False
            )

            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            # ------------------------------------------------------------ #
            #                           LOSS                               #
            # ------------------------------------------------------------ #
            loss = loss_function(predicted_masks, ground_truth_masks.unsqueeze(1))
            batch_losses.append(loss.item())

            # ------------------------------------------------------------ #
            #                       SIGMOID + THRESHOLD                     #
            # ------------------------------------------------------------ #
            sam_prob = torch.sigmoid(predicted_masks).squeeze()
            sam_mask = (sam_prob > 0.5).float()
            sam_mask = sam_mask.unsqueeze(1)

            # ------------------------------------------------------------ #
            #                       METRICS                                 #
            # ------------------------------------------------------------ #
            ious = torch.nanmean(
                compute_iou(sam_mask, ground_truth_masks.unsqueeze(1))
            ).cpu().item()
            dices = torch.nanmean(
                compute_dice(sam_mask, ground_truth_masks.unsqueeze(1))
            ).cpu().item()

            batch_ious.append(ious)
            batch_dices.append(dices)

            # ------------------------------------------------------------ #
            #                     BACKWARD + STEP                           #
            # ------------------------------------------------------------ #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------------------------------------------------------------ #
        # EPOCH TRAIN METRICS
        # ------------------------------------------------------------ #
        train_losses.append(np.nanmean(batch_losses))
        train_loss_std.append(np.nanstd(batch_losses))
        train_ious.append(np.nanmean(batch_ious))
        train_iou_std.append(np.nanstd(batch_ious))
        train_dices.append(np.nanmean(batch_dices))
        train_dice_std.append(np.nanstd(batch_dices))

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        print(f"LR: {current_lr}")
        print(f"Train Loss: {train_losses[-1]:.4f} ± {train_loss_std[-1]:.4f}")
        print(f"Train IoU:  {train_ious[-1]:.4f} ± {train_iou_std[-1]:.4f}")
        print(f"Train Dice: {train_dices[-1]:.4f} ± {train_dice_std[-1]:.4f}")

        # ===================================================================== #
        #                             TEST LOOP                                #
        # ===================================================================== #
        model.eval()
        batch_losses, batch_ious, batch_dices = [], [], []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing"):

                outputs = model(
                    batch["fat"]["pixel_values"].to(device),
                    batch["water"]["pixel_values"].to(device),
                    batch["water"]["input_boxes"].to(device)
                )

                predicted_masks = outputs[0]

                predicted_masks = nn.functional.interpolate(
                    predicted_masks,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False
                )

                ground_truth_masks = batch["ground_truth_mask"].float().to(device)

                loss = loss_function(predicted_masks, ground_truth_masks.unsqueeze(1))
                batch_losses.append(loss.item())

                sam_prob = torch.sigmoid(predicted_masks)
                sam_mask = (sam_prob > 0.5).float()

                iou = torch.nanmean(
                    compute_iou(sam_mask, ground_truth_masks.unsqueeze(1))
                ).cpu().item()
                dice = torch.nanmean(
                    compute_dice(sam_mask, ground_truth_masks.unsqueeze(1))
                ).cpu().item()

                batch_ious.append(iou)
                batch_dices.append(dice)

                scheduler.step(loss.item())

        # ------------------------------------------------------------ #
        # EPOCH TEST METRICS
        # ------------------------------------------------------------ #
        test_losses.append(np.nanmean(batch_losses))
        test_loss_std.append(np.nanstd(batch_losses))
        test_ious.append(np.nanmean(batch_ious))
        test_iou_std.append(np.nanstd(batch_ious))
        test_dices.append(np.nanmean(batch_dices))
        test_dice_std.append(np.nanstd(batch_dices))

        print(f"Test Loss: {test_losses[-1]:.4f} ± {test_loss_std[-1]:.4f}")
        print(f"Test IoU:  {test_ious[-1]:.4f} ± {test_iou_std[-1]:.4f}")
        print(f"Test Dice: {test_dices[-1]:.4f} ± {test_dice_std[-1]:.4f}")

        # ===================================================================== #
        #                           MODEL SAVING                               #
        # ===================================================================== #
        if save and (epoch + 1) % save_every == 0:
            save_dir = f"models/models_{main_model_title}"
            os.makedirs(save_dir, exist_ok=True)

            checkpoint_path = os.path.join(
                save_dir,
                f"fusionSAM_epoch{epoch+1}.pth"
            )

            torch.save(model.state_dict(), checkpoint_path)

            print("\n------------------- MODEL SAVED -------------------\n")

    # ===================================================================== #
    #                         SAVE METRICS TO CSV                            #
    # ===================================================================== #
    save_dir = "Losses, IoUs and Dices"
    os.makedirs(save_dir, exist_ok=True)

    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, num_epochs + 1)),
        "Train_Loss": train_losses,
        "Train_Loss_STD": train_loss_std,
        "Train_IoU": train_ious,
        "Train_IoU_STD": train_iou_std,
        "Train_Dice": train_dices,
        "Train_Dice_STD": train_dice_std,
        "Test_Loss": test_losses,
        "Test_Loss_STD": test_loss_std,
        "Test_IoU": test_ious,
        "Test_IoU_STD": test_iou_std,
        "Test_Dice": test_dices,
        "Test_Dice_STD": test_dice_std
    })

    csv_path = os.path.join(save_dir, f"training_metrics_{main_model_title}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"📊 Metrics saved to: {csv_path}")

    return (
        train_losses, train_ious, train_dices,
        test_losses, test_ious, test_dices,
        train_loss_std, train_iou_std, train_dice_std,
        test_loss_std, test_iou_std, test_dice_std,
        lrs
    )


        