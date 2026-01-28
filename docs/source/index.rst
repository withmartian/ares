ARES Documentation
==================

**ARES (Agentic Research and Evaluation Suite)** is an RL-first framework for training and evaluating code agents.

Unlike traditional frameworks that treat the entire code agent as the optimization target, ARES enables reinforcement learning on the **LLM within the agent**. This provides fine-grained control over long-horizon tasks and opens up new possibilities for mechanistic interpretability research.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   core-concepts
   how-it-works

Getting Started
---------------

See the main `README <https://github.com/withmartian/ares>`_ for installation instructions and quick start examples.

Key Features
------------

* **RL-First Design**: Built around the reinforcement learning loop with observations (LLM requests) and actions (LLM responses)
* **LLM-Level Optimization**: Train the LLM within code agents, not just the agent as a whole
* **Distributed Workloads**: Support for high-volume, distributed training and evaluation
* **Mechanistic Interpretability**: Raw access to LLM requests and responses for deep analysis
* **Async Gym/dm_env like Spec**: Close to Gym/dm_env spec, but incorpirating async methods for performance 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
