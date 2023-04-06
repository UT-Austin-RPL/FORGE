---

layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>

<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>FORGE</title>



<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->

<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>
<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/forge.png">
<link rel="icon" type="image/png" sizes="32x32" href="/forge.png">
<link rel="icon" type="image/png" sizes="16x16" href="/forge.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/forge.svg" color="#5bbad5">

<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->
<link rel="shortcut icon" type="image/x-icon" href="forge.ico">
</head>


<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center><h1><strong><br>Few-View Object Reconstruction <br /> with Unknown Categories and Camera Poses</strong></h1></center>
<center><h2>
    <a href="https://hwjiang1510.github.io/">Hanwen Jiang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://zhenyujiang.me/">Zhenyu Jiang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.cs.utexas.edu/users/grauman/">Kristen Grauman</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a>&nbsp;&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp; 		
    </h2></center>
	<center><h2><a href="https://arxiv.org/pdf/2212.04492">Paper</a> | <a href="https://github.com/UT-Austin-RPL/FORGE">Code (Coming Soon!)</a> </h2></center>




<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
 While object reconstruction has made great strides in recent years, current methods typically require densely captured images and/or known camera poses, and generalize poorly to novel object categories. To step toward object reconstruction in the wild, this work explores reconstructing general real-world objects from a few images without known camera poses or object categories. The crux of our work is solving two fundamental 3D vision problems — shape reconstruction and pose estimation — in a unified approach. Our approach captures the synergies of these two problems: reliable camera pose estimation gives rise to accurate shape reconstruction, and the accurate reconstruction, in turn, induces robust correspondence between different views and facilitates pose estimation. Our method FORGE predicts 3D features from each view and leverages them in conjunction with the input images to establish cross-view correspondence for estimating relative camera poses. The 3D features are then transformed by the estimated poses into a shared space and are fused into a neural radiance field. The reconstruction results are rendered by volume rendering techniques, enabling us to train the model without 3D shape ground-truth. Our experiments show that FORGE reliably reconstructs objects from five views. Our pose estimation method outperforms existing ones by a large margin. The reconstruction results under predicted poses are comparable to the ones using ground-truth poses. The performance on novel testing categories matches the results on categories seen during training.
</p></td></tr></table>
</p>
  </div>
</p>



<h1 align="center">Real-World Demo</h1>
<h2 align="center">Using Online Product Data</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay loop width="100%">
      <source src="./video/online_product_2.mp4"  type="video/mp4">
  </video>
  </td>
      </tr></tbody></table>

<br>

<br>

<h2 align="center">Using iPhone-captured Data</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
      <p align="justify" width="20%"> We tested FORGE on real-world data captured by iPhone. We used three images as inputs of FORGE and compared it with COLMAP which uses dense inputs. FORGE reliably reconstructed the objects from novel categories even though the lighting condition, image capturing stratergy and camera intrinsics are different from training.</p>
  <video muted autoplay loop width="100%">
      <source src="./video/real_2.mp4"  type="video/mp4">
  </video>
</td></tr></tbody></table>

<hr>

<h1 align="center">Problem Definition</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay width="100%">
        <source src="./video/overview2.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
      3D reconstruction is the process of capturing the shape and appearance of real objects<sup><a href="https://en.wikipedia.org/wiki/3D_reconstruction">[1]</a></sup>. Reconstruction results can be represented by explicit 3D representations, e.g., mesh, voxel and point cloud, or implicit representations, like the neural radiance field. In this project, we study the problem of object reconstruction from <b>a few RGB observations</b> with <b>unknown camera poses</b> and <b>unknown object category</b>. Our method Few-view Object Reconstruction that Generalize (FORGE) estimates the relative camera poses of the input views and extractes per-view 3D featrues. The featrues are fused based on the predicted poses to predict a neural volume, encoding the radiance field. The reconstruction results can be produced to 2D by using volume rendering techniques.
</p></td></tr></table>

<br><br>

<hr> <h1 align="center">FORGE Architecture</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/pipeline.png"> <img
src="./src/pipeline.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>
<table width=800px><tr><td> <p align="justify" width="20%">The inputs are a few observations, e.g., five views, of the object. FORGE uses a voxel encoder to extract per-view 3D features, which are defined in their corresponding camera's local frame. Then FORGE predicts the relative camera poses of the input views using the 3D features and 2D raw observations as inputs. The 3D features are transformed into a shared reconstruction space using the rigid transformation computed by the relative camera poses. The features are fused to predict a neural volume that encodes the
radiance field. We use volume rendering techniques to render the reconstruction results. FORGE is trained without 3D shape ground-truth.</p></td></tr></table>
<br>

<hr>


<h1 align="center">Reconstruction Results</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
      <p align="justify" width="20%">To evaluate the generalization ability of FORGE, we propose a new datasets containing both training and novel object categories. The camera poses are randomly sampled. When using ground-truth camera poses, FORGE outperforms previous SOTA PixelNeRF by 2 dB PSNR with 3000 times faster inference speed. We use 5 views as inputs and evaluate the performance on another 5 novel views, where we show 2 of them.</p>
   <a href="./src/gt_compare.png"> <img
src="./src/gt_compare.png" style="width:100%;"> </a>
      </td></tr></tbody></table>

<br>



<h2 align="center">Reconstructing Objects in Training Categories</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
      <p align="justify" width="20%">We show reconstruction results using predicted camera poses on 13 training categories from ShapeNet with 5 input images. FORGE accuratly predicts the shape and appearance of objects.</p>
  <video muted autoplay loop width="100%">
      <source src="./video/train_category_2.mp4"  type="video/mp4">
  </video>
  </td></tr></tbody></table>

<br>

<h2 align="center">Zero-shot Generalization to Novel Categories</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
      <p align="justify" width="20%">We show zero-shot generalization results using predicted camera poses on 10 novel categories from ShapeNet with 5 input images. FORGE reliably reconstructs objects from novel categories with a small PSNR gap of 0.8 dB compared with results on training categories.</p>
  <video muted autoplay loop width="100%">
      <source src="./video/novel_category_2.mp4"  type="video/mp4">
  </video>
  </td>
      </tr></tbody></table>

<br>

<h2 align="center">Zero-shot Generalization to Real Objects</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
      <p align="justify" width="20%">FORGE trained on ShapeNet objects generalize well on real objects from Google Scanned Object dataset. All objects are from novel categories with diverse geometry and texture. FORGE using predicted poses demonstrates better reconstruction quality than PixelNeRF with ground-truth poses.</p>
  <video muted autoplay loop width="100%">
      <source src="./video/gso.mp4"  type="video/mp4">
  </video>
  </td>
      </tr></tbody></table>

<br>

<h2 align="center">Voxel Reconstruction</h2>
<table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle">
    <p align="justify" width="20%">We get voxel reconstruction results by simply thresholding the predicted density of neural volume. The results shows the strong ability of FORGE for capturing 3D geometry even though it is trained without 3D shape ground-truth.</p>
    <a href="./src/vis_voxel.png"> <img
src="./src/vis_voxel.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<br>

<h2 align="center">Pose Estimation Results</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
      <p align="justify" width="20%">We show relative camera pose estimation results on objects from both training and novel categories. Predicted and ground-truth poses are shown in the same color with colored and white faces. FORGE 
achieves 5 degree and 10 degree pose errors on training and novel categories, reducing the errors by more than 60% compared with previous SOTA.</p>
  <video muted autoplay loop width="100%">
      <source src="./video/pose.mp4"  type="video/mp4">
  </video>
  </td>
      </tr></tbody></table>

<br>



<hr>
<!-- <table align=center width=800px> <tr> <td> <left> -->
<center><h1>Citation</h1></center>
<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@article{jiang2022forge,
   title={Few-View Object Reconstruction with Unknown Categories and Camera Poses},
   author={Jiang, Hanwen and Jiang, Zhenyu and Grauman, Kristen and Zhu, Yuke},
   journal={ArXiv},
   year={2022},
   volume={2212.04492}
}
</code></pre>
</left></td></tr></table>


<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>

<center><h1>Acknowledgements</h1></center> 
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->

