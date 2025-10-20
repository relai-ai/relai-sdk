<p align="center">
  <img align="center" src="relai/docs/assets/relai-logo.png" width="460px" />
</p>
<p align="left">
<h1 align="center">RELAI: Simulate → Evaluate → Optimize AI Agents</h1>

**RELAI** is an SDK for building **reliable AI agents**. It streamlines the hardest parts of agent development—**simulation**, **evaluation**, and **optimization**—so you can iterate quickly with confidence.

**What you get**
- **Agent Simulation** — Create full/partial environments, define **LLM personas**, mock **MCP** servers & tools, and generate **synthetic data**. Optionally **condition simulation on real samples** to better match production.
- **Agent Evaluation** — Mix **code-based** and **LLM-based** custom evaluators or use **RELAI platform evaluators**. Turn human reviews into **benchmarks** you can re-run.
- **Agent Optimization (Maestro)** — Holistic optimizer that uses evaluator signals & feedback to improve prompts/configs **and** suggest **graph-level** changes. Also selects **best model/tool/graph** based on observed performance.

## Quickstart

Create a free account and get a RELAI API key: [platform.relai.ai/settings/access/api-keys](https://platform.relai.ai/settings/access/api-keys)

### Installation and Setup

```bash
pip install relai
# or
uv add relai

export RELAI_API_KEY="<RELAI_API_KEY>"
```

## Links

- 📘 **Documentation:** [docs.relai.ai](#)
- 🧪 **Examples:** [relai-sdk/examples](examples)
- 🌐 **Website:** [relai.ai](https://relai.ai)
- 📰 **Maestro Technical Report:** [ArXiV](https://arxiv.org/abs/2509.04642)
- 🌐 **Join the Community:** [Discord](https://discord.gg/sjaHJ34YYE)

## License

Apache 2.0
