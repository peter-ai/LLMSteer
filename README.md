# LLMSteer: The Unreasonable Effectiveness of LLMs for Query Optimization
This repository contains code, embeddings of SQL queries, models, and result artifacts from the preliminary work of LLMSteer. Analysis was done using `python version 3.11.4`. See `requirements.txt` for list of dependencies. Queries were executed using `PostgreSQL version 16.1`.

**Abstract:** Recent work in database query optimization has used complex machine learning strategies, such as customized reinforcement learning schemes. Surprisingly, we show that LLM embeddings of query text contain useful semantic information for query optimization. Specifically, we show that a simple binary classifier deciding between alternative query plans, trained only on a small number of labeled embedded query vectors, can outperform existing heuristic systems. Although we only present some preliminary results, an LLM-powered query optimizer could provide significant benefits, both in terms of performance and simplicity.

Data used in this work is accessible [here](https://www.dropbox.com/scl/fi/o0bqsm5osdm3wsotg7i5e/llmsteer_data.tar.zst?rlkey=8u4t506zpm42vst5w1l4slxca&st=logxe3pp&dl=0) and can be unzipped with the following command: `tar --zstd -xvf llmsteer_data.tar.zst`

The list of hints used in this work originate from [Bao](https://dl.acm.org/doi/pdf/10.1145/3448016.3452838) and can be found in the online appendix [here](https://rmarcus.info/appendix.html)

**Full paper can be found on arXiv [here](https://arxiv.org/abs/2411.02862)**
