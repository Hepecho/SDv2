import argparse, os  # 导入argparse和os模块，用于处理命令行参数和文件操作
import cv2  # 导入OpenCV库，用于图像处理
import torch  # 导入PyTorch库，用于深度学习
import numpy as np  # 导入NumPy库，用于数值计算
from omegaconf import OmegaConf  # 导入OmegaConf库，用于处理配置文件
from PIL import Image  # 导入PIL库，用于图像处理
from tqdm import tqdm, trange  # 导入tqdm库，用于显示进度条
from itertools import islice  # 导入itertools模块中的islice，用于迭代切片
from einops import rearrange  # 导入einops库中的rearrange，用于张量重排
from torchvision.utils import make_grid  # 导入torchvision库中的make_grid，用于生成图像网格
from pytorch_lightning import seed_everything  # 导入pytorch_lightning库中的seed_everything，用于设置随机种子
from torch import autocast  # 导入PyTorch中的autocast，用于自动混合精度
from contextlib import nullcontext  # 导入contextlib中的nullcontext，用于上下文管理
from imwatermark import WatermarkEncoder  # 导入imwatermark库中的WatermarkEncoder，用于添加水印

from ldm.util import instantiate_from_config  # 导入ldm.util中的instantiate_from_config，用于从配置文件中实例化模型
from ldm.models.diffusion.ddim import DDIMSampler  # 导入ldm.models.diffusion.ddim中的DDIMSampler，用于DDIM采样
from ldm.models.diffusion.plms import PLMSSampler  # 导入ldm.models.diffusion.plms中的PLMSSampler，用于PLMS采样
from ldm.models.diffusion.dpm_solver import \
    DPMSolverSampler  # 导入ldm.models.diffusion.dpm_solver中的DPMSolverSampler，用于DPM采样

torch.set_grad_enabled(False)  # 禁用梯度计算


def get_count(outpath, sample_path):
    sample_list = os.listdir(sample_path)
    prefix_list = []
    for sample_name in sample_list:
        prefix_list += sample_name.split('-')[0]
    base_count = len(set(prefix_list))
    grid_list = os.listdir(outpath)
    grid_list.remove('samples')
    prefix_list = []
    for grid_name in grid_list:
        prefix_list += grid_name.split('-')[1]
    grid_count = len(set(prefix_list))
    sample_count = 0  # 初始化样本计数
    return sample_count, base_count, grid_count


def chunk(it, size):  # 定义chunk函数，用于将迭代器按指定大小分块
    it = iter(it)  # 将输入转换为迭代器
    return iter(lambda: tuple(islice(it, size)), ())  # 返回按块大小切片的迭代器


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):  # 定义函数，从配置文件和检查点加载模型
    print(f"Loading model from {ckpt}")  # 打印加载信息
    pl_sd = torch.load(ckpt, map_location="cpu")  # 从检查点加载模型参数
    if "global_step" in pl_sd:  # 如果存在global_step，打印其值
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]  # 提取模型状态字典
    model = instantiate_from_config(config.model)  # 从配置文件实例化模型
    m, u = model.load_state_dict(sd, strict=False)  # 加载状态字典到模型中
    if len(m) > 0 and verbose:  # 打印缺失的键
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:  # 打印意外的键
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):  # 如果使用CUDA，将模型移动到GPU
        model.cuda()
    elif device == torch.device("cpu"):  # 如果使用CPU，将模型移动到CPU
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")  # 报错：设备名称不正确
    model.eval()  # 设置模型为评估模式
    return model  # 返回加载的模型


def parse_args():  # 定义函数，用于解析命令行参数
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    # 新增调试参数
    parser.add_argument(
        "--action",  # 添加action参数，用于指定超参数实验或仅单次推理模式
        type=int,
        default=0,
        help="action mode",
    )
    parser.add_argument(
        "--temperature",  # 添加温度参数，用于指定DDIM的采样温度
        type=float,
        default=1.,
        help="ddim temperature (default temperature=1.0)",
    )
    parser.add_argument(
        "--prompt",  # 添加prompt参数，用于指定生成图像的文本提示
        type=str,
        nargs="?",
        default="a photographic image of a little girl wearing a straw hat",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",  # 添加outdir参数，用于指定结果保存目录
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",  # 添加steps参数，用于指定DDIM采样步数
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",  # 添加plms参数，用于启用PLMS采样
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",  # 添加dpm参数，用于启用DPM采样
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",  # 添加fixed_code参数，是否使用相同的起始代码生成所有样本
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",  # 添加ddim_eta参数，用于指定DDIM的eta值
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",  # 添加n_iter参数，用于指定采样的迭代次数，即指定grid图像的行数
        type=int,
        default=5,
        help="sample this often",
    )
    parser.add_argument(
        "--H",  # 添加H参数，用于指定图像高度（像素）
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",  # 添加W参数，用于指定图像宽度（像素）
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",  # 添加C参数，用于指定潜在空间通道数
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",  # 添加f参数，用于指定下采样因子
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",  # 添加n_samples参数，用于指定每个提示生成的样本数，即批次大小，即grid中列数
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",  # 添加n_rows参数，用于指定网格中的列数，默认为n_samples
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",  # 添加scale参数，用于指定无条件引导的比例
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",  # 添加from-file参数，如果指定，从文件中加载提示
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",  # 添加config参数，用于指定模型配置文件的路径
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",  # 添加ckpt参数，用于指定模型检查点的路径
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",  # 添加seed参数，用于指定随机种子
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",  # 添加precision参数，用于指定评估的精度
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",  # 添加repeat参数，用于指定文件中每个提示的重复次数
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--device",  # 添加device参数，用于指定运行稳定扩散的设备
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cpu"
    )
    parser.add_argument(
        "--torchscript",  # 添加torchscript参数，启用TorchScript
        action='store_true',
        help="Use TorchScript",
    )
    parser.add_argument(
        "--ipex",  # 添加ipex参数，启用Intel IPEX加速
        action='store_true',
        help="Use Intel IPEX",
    )
    parser.add_argument(
        "--bf16",  # 添加bf16参数，启用BF16加速
        action='store_true',
        help="Use bfloat16",
    )
    parser.add_argument(
        "--embedding_path",  # 添加embedding_path参数，指定嵌入路径
        type=str,
        help="Path to a pre-trained embedding manager checkpoint")
    return parser.parse_args()  # 返回解析后的命令行参数


def put_watermark(img, wm_encoder=None):  # 定义函数，用于在图像上添加水印
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # 转换图像颜色空间
        img = wm_encoder.encode(img, 'dwtDct')  # 添加水印
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换图像颜色空间
    return img  # 返回添加水印后的图像


def main(opt):  # 定义主函数
    seed_everything(opt.seed)  # 设置随机种子

    config = OmegaConf.load(f"{opt.config}")  # 加载模型配置文件
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")  # 检查CUDA是否可用，并设置设备
    model = load_model_from_config(config, f"{opt.ckpt}", device)  # 从配置文件和检查点加载模型

    if opt.plms:  # 如果启用PLMS采样，实例化PLMSSampler
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:  # 如果启用DPM采样，实例化DPMSampler
        sampler = DPMSolverSampler(model, device=device)
    else:  # 否则，实例化DDIMSampler
        sampler = DDIMSampler(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)  # 创建输出目录
    outpath = opt.outdir  # 设置输出路径

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"  # 设置水印文本
    wm_encoder = WatermarkEncoder()  # 实例化水印编码器
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))  # 设置水印编码器的水印文本

    batch_size = opt.n_samples  # 设置批次大小
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size  # 设置网格中的行数
    if not opt.from_file:  # 如果未指定提示文件，使用命令行参数中的提示
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:  # 如果指定了提示文件，从文件中加载提示
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(opt.repeat)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")  # 设置样本保存路径
    os.makedirs(sample_path, exist_ok=True)  # 创建样本目录
    sample_count, base_count, grid_count = get_count(outpath, sample_path)
    # sample_count = 0  # 初始化样本计数
    # base_count = len(os.listdir(sample_path))  # 设置基础计数
    # grid_count = len(os.listdir(outpath)) - 1  # 设置网格计数

    start_code = None  # 初始化起始代码
    if opt.fixed_code:  # 如果启用固定代码，设置起始代码
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    if opt.torchscript or opt.ipex:  # 如果启用TorchScript或IPEX优化，进行相应设置
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder
        additional_context = torch.cpu.amp.autocast() if opt.bf16 else nullcontext()
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        if opt.bf16 and not opt.torchscript and not opt.ipex:
            raise ValueError('Bfloat16 is supported only for torchscript+ipex')
        if opt.bf16 and unet.dtype != torch.bfloat16:
            raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                             "you'd like to use bfloat16 with CPU.")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError(
                "Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

        if opt.ipex:  # 如果启用IPEX，进行相应优化
            import intel_extension_for_pytorch as ipex
            bf16_dtype = torch.bfloat16 if opt.bf16 else None
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)

            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if opt.torchscript:  # 如果启用TorchScript，进行相应优化
            with torch.no_grad(), additional_context:
                # 获取UNET脚本化模型
                if unet.use_checkpoint:
                    raise ValueError("Gradient checkpoint won't work with tracing. " +
                                     "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                t_in = torch.ones(2, dtype=torch.int64)
                context = torch.ones(2, 77, 1024, dtype=torch.float32)
                scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                print(type(scripted_unet))
                model.model.scripted_diffusion_model = scripted_unet

                # 获取第一阶段模型解码器脚本化模型
                samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                print(type(scripted_decoder))
                model.first_stage_model.decoder = scripted_decoder

        prompts = data[0]
        print("Running a forward pass to initialize optimizations")
        uc = None
        if opt.scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        with torch.no_grad(), additional_context:
            for _ in range(3):  # 进行3次前向传递以初始化优化
                c = model.get_learned_conditioning(prompts)
            samples_ddim, _ = sampler.sample(S=5,
                                             conditioning=c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=opt.scale,
                                             unconditional_conditioning=uc,
                                             eta=opt.ddim_eta,
                                             x_T=start_code)
            print("Running a forward pass for decoder")
            for _ in range(3):  # 进行3次解码器前向传递
                x_samples_ddim = model.decode_first_stage(samples_ddim)

    precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext  # 设置精度范围
    with torch.no_grad(), \
            precision_scope(opt.device), \
            model.ema_scope():  # 启用EMA范围
        all_samples = list()  # 初始化所有样本的列表
        log_steps = None  # 记录采样中间图像的个数
        for n in trange(opt.n_iter, desc="Sampling"):  # 迭代采样
            for prompts in tqdm(data, desc="data"):
                uc = None
                if opt.scale != 1.0:  # 如果比例不为1.0，设置无条件输入
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)  # 设置条件输入
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]  # 设置生成图像的形状
                samples, intermediates = sampler.sample(S=opt.steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        temperature=opt.temperature,
                                                        x_T=start_code,
                                                        log_every_t=opt.steps // 2)

                # intermediates = {'x_inter': [img], 'pred_x0': [img]}  此处的img是在latent space中d的，大小为(1, 1, 64, 64)
                # 保存采样过程图像变化，每log_every_t保存一次
                if log_steps is None:
                    log_steps = len(intermediates['pred_x0'])
                for sub_index, x_i in enumerate(intermediates['pred_x0']):
                    samples = x_i
                    x_samples = model.decode_first_stage(samples)
                    # 解码第一阶段样本，每一个latent code对应time step个中间图像，表示扩散过程中对应time step的预测t_0y原始图像
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)  # 限制图像值的范围

                    for x_sample in x_samples:  # 保存样本为图像文件
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, f"{base_count:05}-{sub_index:02}.png"))
                        base_count += 1
                        sample_count += 1

                    if sub_index < log_steps - 1:
                        base_count -= len(x_samples)
                        sample_count -= len(x_samples)

                    all_samples.append(x_samples)  # 将样本添加到所有样本的列表中

        # 按log_steps为步长切分all_samples，生成样本网格
        for t in range(log_steps):
            all_samples_ = all_samples[t::log_steps]
            # 生成样本网格并保存
            grid = torch.stack(all_samples_, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)  # 设置每行的图片数量，返回一张网格图像

            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid = put_watermark(grid, wm_encoder)
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}-{t:02}.png'))
            # grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    opt = parse_args()  # 解析命令行参数
    if opt.action == 0:
        main(opt)  # 运行主函数
    elif opt.action == 1:  # 扩散步数
        for s in [15, 25, 35]:
            opt.steps = s
            opt.outdir = f'outputs/txt2img-steps-{s}'
            main(opt)
    elif opt.action == 2:  # 采样温度
        opt.ddim_eta = 0.5  # 默认设置为0则为确定性采样
        for t in [1.5, 2.0, 5.0]:
            opt.temperature = t
            opt.outdir = f'outputs/txt2img-temperature-{t}'
            main(opt)
    elif opt.action == 3:  # DDIM和DDPM采样的插值
        for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            opt.ddim_eta = eta
            opt.outdir = f'outputs/txt2img-eta-{eta}'
            main(opt)
    else:
        pass

