# NVComposer

*Boosting Generative Novel View Synthesis with Multiple Sparse and Unposed Images*

Lingen Li, Zhaoyang Zhang, Yaowei Li, Jiale Xu, Wenbo Hu, Xiaoyu Li, Weihao Cheng, Jinwei Gu, Tianfan Xue, Ying Shan

 <a href='https://lg-li.github.io/project/nvcomposer'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/spaces/TencentARC/NVComposer'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;
 <a href="https://arxiv.org/abs/2412.03517"><img src="https://img.shields.io/static/v1?label=Arxiv Preprint&message=NVComposer&color=red&logo=arxiv"></a>
 
### Abstract 

Recent advancements in generative models have significantly improved novel view synthesis (NVS) from multi-view data. However, existing methods depend on external multi-view alignment processes, such as explicit pose estimation or pre-reconstruction, which limits their flexibility and accessibility, especially when alignment is unstable due to insufficient overlap or occlusions between views. In this paper, we propose NVComposer, a novel approach that eliminates the need for explicit external alignment. NVComposer enables the generative model to implicitly infer spatial and geometric relationships between multiple conditional views by introducing two key components: 1) an image-pose dual-stream diffusion model that simultaneously generates target novel views and condition camera poses, and 2) a geometry-aware feature alignment module that distills geometric priors from dense stereo models during training. Extensive experiments demonstrate that NVComposer achieves state-of-the-art performance in generative multi-view NVS tasks, removing the reliance on external alignment and thus improving model accessibility. Our approach shows substantial improvements in synthesis quality as the number of unposed input views increases, highlighting its potential for more flexible and accessible generative NVS systems.

### To Run

Follow these steps to set up and run NVComposer on your local machine:

- **Clone the Repository**

Download the code from our GitHub repository:
```bash
git clone https://github.com/TencentARC/NVComposer
cd NVComposer
```

- **Set Up the Python Environemnt**

Ensure you have Anaconda or Miniconda installed, then create and activate a Python environment and install required dependencies:

```bash
conda create -n nvcomposer python=3.10
conda activate nvcomposer
pip install -r requirements.txt
```

- **Run the Application**

Launch NVComposer with a single command. The required model checkpoint will be automatically downloaded via `huggingface_hub` and stored in the default cache directory (you can customize this by modifying the `hf_hub_download` parameters):

```bash
python app.py
```

- **Access NVComposer in Your Browser**

Open your browser and go to `http://localhost:7860`.

If you're running the app on a remote server, replace `localhost` with your server's IP address or domain name.

To use a custom port, update the `server_port` parameter in the `demo.launch()` function of app.py.

### Video

[Watch the introduction video](https://lg-li.github.io/project/nvcomposer) in our project page.

[<img src="https://lg-li.github.io/pub-images/li2024nvcomposer-video-cover-2.jpg" width="500">](https://lg-li.github.io/project/nvcomposer)

### Demo

You can [try the demo](https://huggingface.co/spaces/TencentARC/NVComposer) of NVComposer on Hugging Face Space.

### Method

NVComposer contains 1) an image-pose dual-stream diffusion model that generates novel views while implicitly estimating camera poses for conditional images, 
and 2) a geometry-aware feature alignment adapter that uses geometric priors distilled from pretrained dense stereo models.

<img src="https://lg-li.github.io/pub-images/li2024nvcomposer-model.jpg" width="1000">


### Limitation

As a generative model, NVComposer may occasionally produce unexpected outputs. Try adjusting the random seed, sampling steps, or CFG scales to explore different results.
This is the initial beta version of NVComposer, and its generalizability may be limited in certain scenarios. We’re actively working on an improved version with enhanced datasets and a more powerful foundation model.

🤗 We welcome your feedback, questions, or collaboration opportunities. Thank you for trying NVComposer!

### Citation

```
@article{li2024nvcomposer,
  title={NVComposer: Boosting Generative Novel View Synthesis with Multiple Sparse and Unposed Images},
  author={Li, Lingen and Zhang, Zhaoyang and Li, Yaowei and Xu, Jiale and Hu, Wenbo and Li, Xiaoyu and Cheng, Weihao and Gu, Jinwei and Xue, Tianfan and Shan, Ying},
  journal={arXiv preprint arXiv:2412.03517},
  year={2024}
}
```

### License

Please refer to our [license file](https://github.com/TencentARC/NVComposer/blob/main/LICENSE) for more details.