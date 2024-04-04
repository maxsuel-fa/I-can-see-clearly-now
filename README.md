# I Can See Clearly Now

## Results

### Raindrop Removal

The qualitative results for raindrop removal are depicted in Figure 1. The model effectively removes raindrops while preserving image structure and luminosity similar to the ground truth. Quantitatively, metrics such as Structural Similarity Index Measure (SSIM) and Peak Signal-to-Noise Ratio (PSNR) demonstrate significant improvement over raw images, as shown in Table 1.

<figure id="fig:res-dr-1">
    <figcaption>Qualitative results for the raindrop removal task. The first column shows the ground truth image. The second column shows the input image containing raindrops. The third column shows the image generated by the network. The last column represents the difference between the generated and expected images.</figcaption>
    <div style="display: flex; justify-content: center;">
        <div style="flex: 25%; padding: 5px;">
            <img src="images/results_0.png" alt="Ground Truth" style="width:100%">
        </div>
        <div style="flex: 25%; padding: 5px;">
            <img src="images/results_91.png" alt="Input Image" style="width:100%">
        </div>
        <div style="flex: 25%; padding: 5px;">
            <img src="images/results_92.png" alt="Generated Image" style="width:100%">
        </div>
        <div style="flex: 25%; padding: 5px;">
            <img src="images/results_225.png" alt="Difference" style="width:100%">
        </div>
        <div style="flex: 25%; padding: 5px;">
            <img src="images/results_447.png" alt="Difference" style="width:100%">
        </div>
    </div>
</figure>


An analysis of SSIM and PSNR metrics reveals a notable enhancement in image quality compared to raw images. Comparing with the original implementation, our results slightly lag behind due to differences in discriminator network architecture, training routine details, and dataset division.

**Table 1: Quantitative Results for Raindrop Removal**
| Image    | Metric | Value  |
|----------|--------|--------|
| Raw      | PSNR   | 12.8046|
|          | SSIM   | 0.4662 |
| Generated| PSNR   | 20.2564|
|          | SSIM   | 0.7364 |

### Fog Removal

For fog removal, qualitative results vary based on fog density and homogeneity, as seen in Figure 2. While the model excels with less dense and homogenous fog, performance degrades with denser and non-homogeneous fog. Quantitative analysis, shown in Table 2, highlights improved PSNR and SSIM for generated images compared to raw ones, albeit with a smaller improvement compared to raindrop removal.

**Figure 2: Qualitative Results for Fog Removal**
- Column 1: Ground truth image
- Column 2: Input image with fog
- Column 3: Generated image by the model
- Column 4: Difference between generated and ground truth images

Challenges in fog removal stem from the model's original focus on raindrop removal and insufficient training data for fog removal.

**Table 2: Quantitative Results for Fog Removal**
| Image    | Metric | Value  |
|----------|--------|--------|
| Raw      | PSNR   | 11.0827|
|          | SSIM   | 0.4200 |
| Generated| PSNR   | 16.5725|
|          | SSIM   | 0.6597 |

[Insert the last section in English here.]

