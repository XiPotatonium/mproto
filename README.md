# alchemy

希望通过模块化设计将项目编写简化为编写模块+修改配置文件。

## TODO

- [ ] （低优先级）断点续训练。alchemy的断点续训练其实可以有一些假设，例如从某个record dir resume
- [ ] 讨论：需要git集成吗？需要比较工具吗？
- [ ] web server.能不能提供链接的功能，可以在网页上填实验结果的表格，然后将结果链接到某个日志文件夹。是不是其实需要一个.alchemy的文件夹来提供一个全局的信息，包括配置文件，以及实验表格之类的东西，也就是说需要有一个数据库。可以有一个mailbox，跑的实验如果完成了可以有一个提示，如果出错了，可以不可以捕获到异常并提示？（捕获异常似乎是一个比较侵入式的改动，要谨慎考虑，或者修改Runner return的逻辑，默认是捕获异常的，并且将异常塞到Result里面去，这也是具有合理性的，毕竟持续跑实验的时候报错信息很难看到，也并不希望报错干扰其他部分）。

## Known Issues

- [ ] (**重要**) `pytorch 1.11.0 py3.9_cuda11.3_cudnn8.2.0_0`可复现性存在问题。但是`py3.8_cuda11.1_cudnn8.0.5_0`就不存在这个问题。我发现InitEval没有问题，我是把数据集的shuffle关掉的，这说明模型的初始化以及forward是没有问题的。tagging模型是可复现的，说明框架上是没有问题的。甚至`use_determininstic_algorithms`检查是可以通过的，所以就很迷惑了。可能是`TrfEncoder`存在一些问题，如果设置`use_lstm=true`就不可复现了(似乎单层LSTM也不会出错)，在`BiPyramid`和`PnRNet`上都是这个样子，`BertBPETagging`因为是没有`TrfEncoder`所以是可复现的
- [x] `pytorch1.8`的`AdamW`为什么`betas`会有问题？只有PnR会有问题，别的模型可以用`AdamW` [github issue](https://github.com/pytorch/pytorch/issues/53354)

## 依赖

conda:

* rich
* loguru
* typer
* tomlkit
* pytorch
* numpy
* tensorboard (optional)

pip:

* fastapi (optional)
* uvicorn (optional)
* pynvml (如果用户自定义entry且不使用官方的alloc_cuda，那么不是必须的)

## Config

因为做实验，特别是调参的时候，是需要快速迭代配置的。 因此像hydra那样的过于规整的多配置管理似乎不太合适。还是考虑全部写在一个文件里，仅通过文件内的复制粘贴就可以实现多配置。

### Json配置文件

PROS:

* 比较简单
* 有类型系统
* python内建支持且有比较好的prettify

CONS:

* 一大坨
* 没有比较复杂的一些东西（例如无穷大，时间之类的，不过可以通过另外的方式解决且这些需求不常见）
* 不支持注释


### TOML配置文件

PROS：

* 比较简单
* 有类型系统

CONS：

* 字典类型(花括号)只能单行，有点笨
* 没有none类型
* `tomlkit`库可以做到很好的`load`和`store`，但是对于修改后的配置文件的保存可能有问题，容易生成无法被再次自动读取的配置文件(tomlkit会修复吗)

### YAML配置文件

PROS:

* 功能非常多

CONS:

* 太复杂
* 缩进是异端

## SymbolTbl

alchemy库存在一个全局的符号表，符号表模拟编程语言的行为。理论上可以通过符号表访问一切。

**注意**: 如果在Runner运行完成之后还有后续的逻辑（特别是还需要占用GPU的工作），需要在运行完成之后手动将符号表reset，否则符号表中的模型、task等可能不会释放占用的GPU和内存资源

**注意**：一般而言不要在类中将符号表中的数据作为成员变量，**特别是内建成员**，因为内建成员一般而言不允许多进程（可能导致复制数据集、或者其他一些未经测试的错误），将其作为成员变量，可能在多进程时会产生问题。
建议只通过符号表来访问内建成员。（例如在pytorch的多进程Dataloader中）内建成员一般只会在主进程中被赋值，加载进程中不会有多余的内建成员，这样可以减少不必要的内存消耗。

访问符号表，区分：

* set（设置值，如果键冲突，那么报错）
* try_set（设置值，如果键冲突，返回False）
* force_set（设置值，如果键冲突，覆盖）
* get（得到值，如果不存在，那么报错）
* try_get（得到值，如果不存在，那么返回default）

符号表包含三个部分，内建成员、全局符号表和局部符号表（栈）

### 内建成员

* registry
* plugins

* task
* model
* optimizer
* train_sched

* cfg: 配置，一般是`TOMLDocument`
* device_info: 字典形式的device信息
* device: torch形式的device信息，可以用于`model.to(device)`
* console: 控制台输出

### 全局符号表

可以将一些自定义的组件（例如tensorboard，accelerator），数据（例如debug需要输出的数据，case study需要输出的数据），变量（例如是否保存评估结果）保存在全局符号表


### 局部符号表（栈）

堆栈结构，类似编程语言中的函数局部变量，一般而言比较少用到

可以创建栈帧，通过API仅能访问栈顶符号表的变量。

## AlchemyRunner

对运行逻辑（例如训练、评估、推理）的封装，可以继承以实现自己的运行逻辑

## RunnerPlugin

alchemy可以添加插件，以提供不同的功能

可以通过配置文件的`plugins`设置。一个例子：

```toml
plugins = [
    { type = "alchemy.plugins.BasicSetup" },
    { type = "alchemy.plugins.FileLogger", log_dir = "records/weixin/tagger/train", subdirs = ["detail_log"] },
    { type = "alchemy.plugins.Backup", paths = ["alchemy", "src"] },
    { type = "alchemy.plugins.TensorboardLogger" },
    { type = "alchemy.plugins.Seeding", seed = 0, use_deterministic_algorithms = true },
    { type = "alchemy.plugins.DisplayRunningInfo" },
]
```

### BasicSetup

一些基本设置，例如debug，错误显示，控制台日志

### FileLogger

生成日志文件夹。设置文件logger，生成`README.md`并将配置文件保存为`cfg.toml`。
如果没有关闭文件（控制台输入`--no-file`），会在runner.data中添加`record_dir`和`checkpt_dir`（`pathlib.Path`类型），
分别是日志文件夹以及模型缓存文件夹（一般是`record_dir/checkpt`）的路径。

**注意**：如果不想生成日志文件夹，不推荐通过改配置文件的方式（即从配置的`plugins`中删除`alchemy.plugins.FileLogger`），推荐传控制台参数`--no-file`来disable日志文件夹生成。因为生不生成日志是单次实验特定的设置（也就是说不希望单次设置多次使用的设置，其他的例如`device`），单次实验特定的设置应该从控制台传入。

### Backup

拷贝文件，一般而言需要拷贝`alchemy`和`src`（用户文件夹）

### TensorboardLogger

初始化tensorboard

### Seeding

设置种子并进行可复现设置

### DisplayRunningInfo

显示本次任务的一些信息，例如配置文件中的`tag`；硬件和驱动信息`device`（包含GPU类型和驱动版本）；PyTorch的版本；配置文件

## AlchemyTask

定义了一个任务，例如命名实体识别任务。什么是一个任务？一个任务需要有类似的元信息（例如NER的元信息就是实体类别）和统一的评估方法。

`AlchemyTask`需要负责加载数据集并维护一些加载数据集和评估时需要的元数据（例如分类类别，word2vec, tokenizer）

全局唯一，一定会生成

## AlchemyModel

模型。注意`AlchemyModel`实际上是一个wrapper。用户应该先编写pytorch的模型(nn.Module)或者huggingface transformers库的模型。`AlchemyModel`在初始化的时候应该将`self._model`设置为pytorch或者transformers的模型。

`AlchemyModel.forward`将会负责将输入转发到`self.model.forward()`中，并且处理返回值，如果`needs_loss`，那么需要计算loss并将loss（`float`形式而不是`torch.Tensor`）与模型的原始output一起返回（推荐做法是一起形成一个dict，并且如果`needs_loss`那么`outputs["loss"] = loss`）。
`OutputPipeline`的输入就是`AlchemyModel.forward`的返回值。

计算损失的相关配置的设置位置遵照`AlchemyModel.criterion_cfg`。计算损失的配置和模型的配置需要严格区分开，在不需要计算损失的场景下（例如inference），这部分配置是可以删掉的。

如果`requires_grad`，那么需要计算loss的grad并优化模型（包括`zero_grad`，`loss.backward`，以及`optimizer.step`和`sched.step_lr`）。
可以将计算loss的grad并优化模型的这四步（`zero_grad`，`loss.backward`，以及`optimizer.step`和`sched.step_lr`）编写到模块`BackwardHandler`中。
Alchemy提供了一个默认的`BackwardHandler`，`DefaultBackwardHandler`包含默认的上述功能，并且实现了梯度累计的功能（步长为1也就是每个step都更新参数）。
在debug模式下（`data["debug] == True`），还提供了loss的命名以及额外的报错信息（具体见实现）。

**DISCUSSION**: 为什么`forward`还需要负责`zero_grad`，`loss.backward`，以及`optimizer.step`和`sched.step_lr`？因为梯度回传存在很多做法，例如梯度累加，或者分布式的时候可能有不同的backward函数。
因此Alchemy设计了`BackwardHandler`模块。具体配置文件的设置位置遵照`AlchemyModel.backward_cfg`

全局唯一，一定会生成

## AlchemyOptimizer

优化器。也是一个wrapper。需要在构造的时候设置`self.optimizer`（根据配置文件初始化一种`torch.optim.Optimizer`）

全局唯一，模型训练的时候需要，但是inference的时候可以不生成

## AlchemyTrainScheduler

调度器。注意调度器有两个方面的职责，一个方面是记录现在是第几个*step*，第几个*epoch*，是当前*epoch*中的第几个*step*，并且负责在epoch开始/epoch结束/step开始/step结束的时候调用对应的pipeline（这些pipeline也可以被理解为定时回调函数）。注意是**Train**Scheduler，因此这里的*step*指的是train step，不包括eval step。

另一个职责是作为一个wrapper，调度学习率，推荐学习率调度的单位时间是*step*而不是*epoch*。需要在构造的时候根据配置文件初始化`self.lr_sched`（一般而言是`torch.optim.lr_scheduler._LRScheduler`）

alchemy提供了一个`alchemy.sched.NoamScheduler`

全局唯一，模型训练的时候需要，但是inference的时候可以不生成（所以在编写模块的时候需要考虑到inference时候是不应该访问`cur_step`之类的信息的）。

## 一些Pipeline

alchemy使用pipeline来表示一些需要被顺序执行的代码块。

### Pipeline for AlchemyTrainScheduler

`AlchemyTrainScheduler`会在epoch开始，epoch结束，step开始，step结束调用配置中设定的pipeline，这些pipeline可以被理解为是一系列定时回调。

一个比较重要的功能是需要在这些回调中进行评估，可以设定是多少个step评估一次或者多少个epoch评估一次，例如：

```python
@SchedPipeline.register("EvalEpochEEPipeline")
class EvalEpochEEPipeline(EndEpochPipeline):
    def __init__(self, period: int, split: str = "dev", **kwargs) -> None:
        super().__init__()
        self.period = period
        self.split = split

    def __call__(self, **kwargs) -> Dict[str, Any]:
        cur_epoch = sym_tbl().train_sched.cur_epoch
        if (cur_epoch + 1) % self.period == 0:
            logger.info(
                "EvalEpoch at epoch {} step {}".format(
                    sym_tbl().train_sched.cur_epoch,
                    sym_tbl().train_sched.cur_step
                )
            )
            evaluate(self.split)
        return kwargs
```

### DataPipeline

`DataPipeline`负责控制数据的加载。设计参考了`torchdata`库，将数据的加载过程拆分成流水线的过程以方便调整，一个配置文件的例子：

```toml
pipes = [
    { type = "alchemy.pipeline.lst.SequenceWrapper", datapipe = ["/home/wsh/data/datasets/ace04/train.jsonl"] },
    { type = "src.task.ner.datapipe.JsonLOpener" },
    { type = "alchemy.pipeline.itr.SplitByWorker" },
    { type = "alchemy.pipeline.itr.Shuffle" },
    { type = "src.task.ner.datapipe.ParseJsonDoc" },
    { type = "src.task.ner.datapipe.PruneLongText" },
    { type = "src.task.ner.datapipe.Sample2Encoding" },
    { type = "alchemy.pipeline.itr.Batch", batch_size = 8 },
]
```

**NOTE**:

* 注意不再推荐在Dataloader中进行batching，为了效率考虑，可以用`alchemy.pipeline.Batch`把batching做好，注意因为在多线程中`torch.Tensor`是无法传递的，所以为了可扩展性推荐在pipeline中处理的都应该是`np.ndarray`，到`collate_fn`的时候再变成`torch.Tensor`。
* `alchemy.pipeline.itr.*`中都是`IterableDataset`，`alchemy.pipeline.lst.*`都是`Dataset`，`alchemy.pipeline.lst.ItrToLst`可以将`IterableDataset`的数据转化成`Dataset`，这样在数据集不大且需要很多epoch的情况下效率比较高。
* 如果最终的数据集是`IterableDataset`，那么不能设置`shuffle = true`，需要使用`alchemy.pipeline.itr.Shuffle`来进行shuffle.
* 如果要做多线程加载，需要`alchemy.pipeline.itr.SplitByWorker`来进行sharding
* 因为`Datapipe`也许会在多线程中运行（设置了`num_workers`），因此在构造的时候不要将过于复杂的对象设置为成员

关于`Dataloader`构造函数中的`collate_fn`：`alchemy`库不会进行任何设置，如果有统一的`collate_fn`，可以在`AlchemyTask.load_dataset`的时候设置

```python
def MyTask(AlchemyTask):
    def load_dataset(self, split: str, **kwargs):
        ...
        self._datasets[split] = (datapipe, {"collate_fn": collate_fn, **kwargs})
```

或者也可以自己写一个`Registrable`的`CollateFn`模块，方便使用配置文件灵活调整

**DISCUSSION**: 为什么`CollateFn`不作为`alchemy`的模块？因为不同的应用的data会有很大的格式差别，用户自己决定就好。

### OutputPipeline

负责处理输出，将输出转化为适合评估或者导出的格式。如果同一个任务的不同模型有不同的输出格式（例如生成式抽取模型的不同生成格式），那么可以定义一些`OutputPipeline`将这些不同的输出转化为同一种格式。

`AlchemyTask.step`中是不需要做`OutpuPipeline`的，`eval_step`才需要，因为大部分情况下（除非有特殊原因需要生成训练数据的输出），训练是不需要这些格式化的结果的。

### EvalPipeline

`EvalPipeline`用于评估。为什么评估是一个Pipeline？因为在评估的同时还会做一些其他的操作，例如输出日志，检查是否是最佳，以及保存checkpoint，使用Pipeline的形式可以方便地选择性开关。

`EvalPipeline`在`AlchemyTask.end_eval`中被调用，注意，`EvalPipeline.begin_eval`会在`AlchemyTask.begin_eval`中被调用，可以用于在`AlchemyRunner.data`中先定义一些变量，让`AlchemyTask.eval`的时候可以将eval数据集的结果保存（在`alchemy.pipeline.output.Collect`实现）以便评估

## NOTE

### DEBUG

可以考虑先使用静态分析工具`mypy`。

官方的程序入口包含控制台参数`--debug`，可以设置是否使用debug。在`alchemy.plugins.BasicSetup`中，如果`running_context.data["debug"] == True`，会设置`CUDA_LAUNCH_BLOCKING=1`保证GPU报错位置准确，控制台logger的level会从默认的`INFO`调整到`DEBUG`。在`alchemy.plugins.FileLogger`中，如果`running_context.data["debug"] == True`，那么文件日志的level会从默认的`INFO`调整到`DEBUG`。

另一个设置是关于是否生成record文件夹。官方的alchemy入口包含控制台参数`--no-file`。在`alchemy.plugins.FileLogger`中，如果`running_context.data["no_file"] == True`，那么不生成record_dir

### Reproducibility

关于可复现性，pytorch 1.9新增了`use_determininstic_algorithms`，可以在配置文件中设置`use_determininstic_algorithms = true`来打开

### 防呆设计

半夜跑代码迷迷糊糊会把一些东西写错，特别是config。需要非常强健的防呆设计。

* `alchemy.util.filter_optional_cfg`做了一个传参的封装，如果配置文件中没有指定，那么会使用函数中设定的默认值，并且会显示warning，便于调试

### DistributedDataParallel

目前已经确定可以使用`huggingface`的`accelerate`（在`alchemy.util.ddp.accelerate`下）：

* 需要覆盖`alchemy.plugins.BasicSetup`，包含构建`Accelerator`并且控制只有`accelerator.is_local_main_process`才能做控制台输出。
* 需要使用`DataPipeline`中需要使用sharding，保证每个node的数据是不一样的，也可以在`accelerator.prepare`中设置dataloader而不用手动sharding（accelerate会同时同步随机数以及做batching，所以似乎要保证pipeline中传递的data都是tensor，以保证是可以自动batching的）。
* 需要一个`AccelerateOptimizer`因为`Accelerate`要求使用`accelerator.clip_grad_norm_`来代替`torch`的`torch.nn.utils.clip_grad_norm_`。
* 需要写一个`BackwardHandler`，使用`accelerator.backward(loss)`来代替`loss.backward`，注意`BackwardHandler`并不是`alchemy`官方的部件。
* `EvalPipeline`需要注意eval之前需要`accelerator.wait_for_everyone()`保证每个线程都训练完了，只有主线程(`accelerator.is_main_process`)才需要做eval和logging，另外保存模型的时候需要先`accelerator.unwrap_model(model)`
* `run_task_fn`需要做对应修改以初始化accelerate的内容

### Case Study

case study是一个很重要的东西。不仅在最终的论文阶段，在模型的早期开发调试阶段也有很多用处

一些模型中间值（例如attention）可以保存在符号表中，并在`EndStepPipeline`中保存（建议自己编写一个`EndStepPipelineForCaseStudy`）

### Records

#### 需要能够输出

* 代码的备份。用git有点大炮打蚊子的感觉（在大规模调参的时候用git并不合适，并且我认为版本控制需要人工，不能自动化），反正也没有多少大小，直接把代码复制一份。注意alchemy这些基本内容也要复制，需要保证复制回来能直接重现.`alchemy.plugins.Backup`提供了这个功能
* 虽然代码备份用git大炮打蚊子，但是不得不说有的时候还是需要类似git的比较功能的，可以去看一下python标准库中的difflib,filecmp,dircmp这些库，可能可以搓一个相对简单的脚本实现类似git的功能。
* 参数的备份。将参数导出。`alchemy.plugins.FileLogger`提供了这个功能
* tensorboard。`alchemy.plugins.TensorboardLogger`提供了相关初始化

#### 需要能够简单地编辑实验的目的并且方便检索

这也是一个防呆设计，为了防止大规模调参的时候第二天起来忘记这个实验跑的是什么了。

`alchemy.plugins.FileLogger`会自动在record文件夹中生成一个`README.md`。配置文件中需要写明`tag`，将会变成`README.md`的标题，官方的入口还支持传入控制台参数`--desc`，将会变成`README.md`的内容

最好还是有一个前端，跑一个服务器

### 如何train-and-test

alchemy官方并不提供train and test的入口，但是实现了`alchemy.runner.ALchemyTester`。这个Runner不会进行训练，而是直接进行eval，并且不会读取`AlchemyTrainScheduler`和`AlchemyOptimizer`（因此在编写评估代码的时候需要避免使用这两个模块，或者单独为test评估编写一个不需要用到这两个模块的评估代码）

不提供官方入口是因为往往需要依据具体的任务修改训练配置文件，以使得这个配置文件可以直接用于测试。下面展示一个样例：

```python
def train_and_test(
    cfgs: List[tomlkit.TOMLDocument],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
):
    run_results = run(
        cfgs=cfgs,
        device=device,
        user_dir=user_dir,
        desc=desc,
        debug=debug,
    )

    test_cfgs = []
    # 因为return返回的顺序可能和cfg不同，所以这里再次从record_dir中读取cfg
    for run_result in run_results:
        if isinstance(run_result, Exception):
            # 遇到exception
            raise run_result
        cfg = prepare_cfg(run_result.record_dir / "cfg.toml")

        cfg["runner"] = "alchemy.runner.Tester"     # 将runner指定为Tester

        # test不需要sched和optim
        if "sched" in cfg:
            cfg.pop("sched")
        if "optim" in cfg:
            cfg.pop("optim")

        if "train" in cfg["task"]["datasets"]:
            cfg["task"]["datasets"].pop("train")
        if "dev" in cfg["task"]["datasets"]:
            cfg["task"]["datasets"].pop("dev")
        # "test" is required
        assert "test" in cfg["task"]["datasets"]

        cfg["model"]["model_path"] = str(run_result.record_dir / "checkpt" / "best")

        # 因为下面这部分不通用，所以train-and-test不考虑作为一个alchemy的官方入口
        for plugin in cfg["plugins"]:
            if plugin["type"] == "alchemy.plugins.FileLogger":
                train_log_dir = Path(plugin["log_dir"])
                test_log_dir = train_log_dir.with_name("test")
                plugin["log_dir"] = str(test_log_dir)

        evalpipes = cfg["task"]["evalpipes"]
        for evalpipe in list(evalpipes):        # 浅拷贝一份，防止remove in iter
            if evalpipe["type"] == "src.task.ner.evalpipe.EvalNer":
                evalpipe["store_history"] = False
            elif evalpipe["type"] == "src.task.ner.evalpipe.LogBest":
                # 保留log best，可以打印best_info
                pass
            elif evalpipe["type"] == "src.task.ner.evalpipe.SaveStepExamples":
                evalpipe["type"] = "src.task.ner.evalpipe.SaveExamples"
            else:
                evalpipes.remove(evalpipe)

        test_cfgs.append(cfg)

    return run(
        cfgs=test_cfgs,
        device=device,
        user_dir=user_dir,
        desc=desc,
        debug=debug,
    )
```
