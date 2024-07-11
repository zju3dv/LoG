## Differences from Original Version

1. **Removed the 0.3 Pixel Addition in 2D Filtering**
   - The original version added an extra 0.3 pixels to 2D filtering operations. This fork removes that addition for more precise filtering.

2. **Added Functionality to Return ID of the Point Contributing Most to Each Pixel**
   - This fork introduces the ability to track and return the ID of the point that has the largest contribution to each pixel, aiding in detailed analysis.

3. **Added Feature to Return Maximum Weight Value for Each Point in the Rendered Image**
   - Now, it's possible to get the maximum weight value of each point in the image, useful for understanding point distribution and impact.

---


# Differential Gaussian Rasterization

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
