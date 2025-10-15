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
from monai.metrics import compute_iou
from monai.metrics import compute_dice

def train_test_model (optimizer : torch.optim.Optimizer,
                      loss_function : torch.nn.Module,
                      num_epochs : int,
                      train_dataloader : torch.utils.data.DataLoader,
                      test_dataloader : torch.utils.data.DataLoader,
                      model : torch.nn.Module,
                      device : str,
                      target_width : int,
                      target_height : int,
                      save=False) :

    model.to(device)
    model.train()

    train_losses = []
    train_ious = []
    train_dices = []
    test_losses = []
    test_ious = []
    test_dices = []

    for epoch in range(num_epochs):

        print(f'EPOCH: {epoch+1}')

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
            batch_ious.append(ious.mean())
            batch_dices.append(dices.mean())

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mean_train_loss = mean(batch_losses)
        train_losses.append(mean_train_loss)
        print(f'Mean Train Focal loss: {mean_train_loss}')

        mean_train_iou = mean([t.cpu().item() for t in batch_ious])
        train_ious.append(mean_train_iou)
        print(f'Mean Train IoU: {mean_train_iou}')

        mean_train_dice = mean([t.cpu().item() for t in batch_dices])
        train_ious.append(mean_train_dice)
        print(f'Mean Train Dice: {mean_train_dice}')

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

            batch_ious.append(iou.cpu().item())
            batch_dices.append(dice.cpu().item())

        mean_test_loss = np.nanmean(batch_losses)
        test_losses.append(mean_test_loss)
        # print(f'Mean Test Focal loss: {mean_test_loss}')

        if (len(batch_ious) >= 1) :
            mean_test_iou = np.nanmean(batch_ious)
        else :
            mean_test_iou = np.nan
        test_ious.append(mean_test_iou)
        print(f'Mean Test IoU: {mean_test_iou}')

        if (len(batch_dices) >= 1) :
            mean_test_dice = np.nanmean(batch_dices)
        else :
            mean_test_dice = np.nan
        test_dices.append(mean_test_dice)
        print(f'Mean Test Dice: {mean_test_dice}')

        #########################################
        ############## MODEL SAVING #############
        #########################################

        if (save) :
            if (epoch%2 == 0) :
                # Specify the file path where you want to save the model parameters
                checkpoint_path = f'models/sam_huge_first{epoch+1}.pth'

                # Save the parameters of the entire model
                torch.save(model.state_dict(), checkpoint_path)
                print("----------------------------------------------------")
                print("------------------- Model Saved! -------------------")
                print("----------------------------------------------------")

        model.train()
    return train_losses, train_ious, train_dices, test_losses, test_ious, test_dices

def test_with_visualization (test_dataloader : torch.utils.data.DataLoader,
                             ds : pd.DataFrame,
                             model : torch.nn.Module,
                             device : str,
                             target_width : int,
                             target_height : int,
                             save=False,
                             visualize = True) :
    test_ious = []
    iou_results = {}
    model.eval()

    # Iteratire through test images
    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            # forward pass
            outputs = model(pixel_values=batch["pixel_values"].cuda(),
                            input_boxes=batch["input_boxes"].cuda(),
                            multimask_output=False)

            ground_truth_masks = batch["ground_truth_mask"].float().cuda()

            # apply sigmoid
            sam_mask_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            sam_mask_prob = sam_mask_prob.cpu().numpy().squeeze()
            sam_mask = (sam_mask_prob > 0.5).astype(np.uint8)

            sam_mask = torch.tensor(sam_mask, device = device).unsqueeze(0).unsqueeze(0)

            sam_mask = nn.functional.interpolate(sam_mask,
                    size=(target_height, target_width),
                    mode = 'nearest')

            # print(sam_mask.shape)
            # print(ground_truth_masks.shape)

            iou = compute_iou(sam_mask,
                            ground_truth_masks.unsqueeze(1))

            sam_mask = sam_mask.squeeze(0).squeeze(0)

            print(f'IoU: {iou}')
            test_ious.append(iou)

            print(f"saving iou : {float(iou.squeeze())}")

            # Inside the loop:
            scan_filename = os.path.basename(batch["filename"][0])  # e.g., 13_000_w.png

            # Look up the corresponding mask filename in ds
            mask_row = ds[ds['image_path'].str.endswith(scan_filename)]
            if len(mask_row) == 1:
                mask_filename = os.path.basename(mask_row['mask_path'].values[0])  # e.g., 13_000_mask.png
                iou_results[mask_filename] = float(iou.squeeze())

            # ======================
            # 🔹 SAVE PREDICTED MASK
            # ======================
            if save:
                mask_path = batch["filename"][0]  # should point to the scan file

                # Extract subject and week from path
                # Example: data/8.331/T3-21J/Water/13_000_w.png
                parts = os.path.normpath(mask_path).split(os.sep)
                try:
                    subject = parts[-4]  # e.g. "8.331"
                    week = parts[-3]     # e.g. "T3-21J"
                except IndexError:
                    subject, week = "Unknown", "Unknown"

                # Derive prediction filename: remove '_w' and append '_mask_pred'
                mask_filename = os.path.basename(mask_path)       # e.g. 13_000_w.png
                pred_name = mask_filename.replace("_w", "")       # 13_000.png
                pred_name = pred_name.replace(".png", "_mask_pred.png")  # 13_000_mask_pred.png

                # Convert mask to uint8 [0–255]
                sam_mask_np = sam_mask.squeeze().detach().cpu().numpy().astype(np.uint8) * 255

                # Create directory: predictions/<subject>/<week>/Predictions/
                save_dir = os.path.join("predictions", str(subject), str(week), "Predictions")
                os.makedirs(save_dir, exist_ok=True)

                # Final path
                save_path = os.path.join(save_dir, pred_name)

                # Save mask
                cv2.imwrite(save_path, sam_mask_np)

            if (visualize) :
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.imshow(batch["pixel_values"][0,1], cmap='gray')
                plt.title('Zebrafish Scan')
                plt.axis('off')

                plt.subplot(1,3,2)
                plt.imshow(batch["ground_truth_mask"][0], cmap='copper')
                plt.title('Actual Mask')
                plt.axis('off')

                plt.subplot(1,3,3)
                plt.imshow(sam_mask.cpu(), cmap='copper')
                plt.title('Predicted Mask')
                plt.axis('off')

                plt.tight_layout()
                plt.show()

    # Use the mask filename from your dataset as key
    ds = ds.copy()
    ds["IoU"] = ds["mask_path"].apply(
        lambda p: iou_results.get(os.path.basename(p), np.nan)
    )

    return ds
        