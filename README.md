# ChannelPatient

**把患者当作一条"未标定的信道"的患者模拟器。**

真实问诊里，医生听的不只是"这句话"，还有"这个人"：有的患者夸大，有的淡化，有的说不清楚，有的要追问几遍才说，有的一开口就带着自己（错误）的诊断，有的把上次医生的话转述错了。好医生一边推断病情，一边推断这个人的报告习惯，并据此给每句话打折。现在的 LLM "医生"往往把每句话都照单全收。

ChannelPatient 用来把这个差距变成可测的量。它是一个更大课题（交互式医疗 LLM 中的 state-channel confounding）的患者侧。

## 仓库里有什么

- **带显式失真信道的患者模拟器。** 病例的每一条临床事实都有*真值*和*报告值*。报告值由规则化的信道产生（夸大、模糊、漏报、自诊；淡化与转述失真在计划中），每型有数值参数。说不说、说多少由代码决定，语言模型只负责把已定的内容说成人话。性格与说话风格是独立的一层，沿用 *Patients With Personality* 的 HEXACO 人格设计。
- **一个 oracle。** 因为失真规则是显式的，每一轮都能精确算出（诊断，信道）的联合后验，包括在某条事实被记录核实后对之前的陈述重新解码。任何医生模型都可以用它做参照。
- **问诊环境和医生接口。** 医生只看得到一句简介和对话记录，通过一个小协议行动：提问、查记录、问家属、开检查、下诊断。任何医生——提示词驱动的 LLM、脚本提问者、或维护信念的 agent——都通过同一个两方法接口接入。
- **一个胸痛簇**：五个鉴别诊断、槽位级的事实表 schema、医学似然表、由 DDXPlus 和手写构造的示例病例。
- **保真度检查与审计**：按规则重算每一条报告值；预注册的门槛覆盖泄露、一致性、抗诱导、分型可辨性、规则遵从。
- **stub 模式**：整条流水线可以不调用任何 LLM 跑通（关键词分类、模板措辞、正则核对），没有 API key 也能测试和复现逻辑。

## 快速开始

```bash
pip install -r requirements.txt
pytest tests -q                                   # 不调用 API

python run_episode.py --config configs/m1_stub.yaml        # stub 运行：11 例 × 4 信道 × 2 种子
python -m scc.sim.audit results/episodes/m1_stub           # 按规则重算每条报告值
python -m scc.analysis.trajectory results/episodes/m1_stub # 信念轨迹 vs oracle

cp .env.example .env                              # OpenAI 兼容接口的地址与 key，只有 LLM 措辞需要
python run_episode.py --config configs/m2_smoke.yaml
```

自己当医生，一次一轮：

```bash
python scripts/doctor_turn.py --session A --start cp_001 "exaggerate:k=2,flooding_rate=0.3"
python scripts/doctor_turn.py --session A "Where exactly is the pain?"
python scripts/doctor_turn.py --session A "VERIFY RECORD: prior_severity"
python scripts/doctor_turn.py --session A "DIAGNOSIS READY: GERD"
```

## 外部资源（不随仓库分发）

| 资源 | 用途 | 获取方式 |
|---|---|---|
| DDXPlus 测试集 | 病例骨架和似然表初值 | `figshare.com/articles/dataset/DDXPlus_Dataset/20043374`（CC BY 4.0）→ `data/external/ddxplus/` |
| PatientsWithPersonality | 角色卡提示词与评测函数（仅 PWPRole 需要） | `git clone https://github.com/mo374z/PatientsWithPersonality external/PatientsWithPersonality`（commit 5357a22） |
| AgentClinic | 对比实验用的 OSCE 病例格式 | `git clone https://github.com/SamuelSchmidgall/AgentClinic external/AgentClinic`（commit b6570ed） |
| IMCS-21、CPB-Bench | 校准统计与 few-shot 风格句 | 按各自许可自行获取；`scc/data/mine_style.py` 重新生成 `data/style/` |

许可见 `data/MANIFEST.md`，数据结构说明见 `external/notes/`。

## 目录

```
scc/types.py         数据结构与医生 <-> 环境契约
scc/env/             病例、信道核、似然表、oracle；clusters/chest_pain/
scc/sim/             模拟器组件、环境、脚本医生、保真度门槛
scc/doctor/base.py   Doctor 协议、动作解析、给医生提示词用的动作格式文本
scc/analysis/        轨迹、指标、报告
scc/data/            DDXPlus 探索与建病例、语料挖掘
tests/               单元与集成测试（stub 模式，不调 API）
configs/ prereg/     运行配置、预注册门槛
```

用法和接入医生的说明见 `scc/README.md`。

## 许可

MIT。上游组件保留各自许可：PatientsWithPersonality 与 AgentClinic 为 MIT；DDXPlus 为 CC BY 4.0；CPB-Bench 为 CC BY-NC-ND 4.0，本仓库只用其统计量，不再分发其数据。
