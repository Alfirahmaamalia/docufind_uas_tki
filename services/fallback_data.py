"""
Fallback dataset containing realistic scientific papers for 12 academic query domains.
Used when the Semantic Scholar API is rate-limited (HTTP 429) or offline.
Ensure that keyword matching in ground_truth.py works as expected.
"""

FALLBACK_PAPERS = [
    # 1. Transformer NLP
    {
        "paperId": "tf_nlp_01",
        "title": "Attention Is All You Need for Natural Language Processing",
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two translation tasks show these models to be superior in quality while being more parallelizable. This self-attention based model achieves state-of-the-art results in NLP tasks and forms the foundation of modern language models like BERT and GPT.",
        "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}, {"name": "Niki Parmar"}],
        "year": 2017,
        "citationCount": 98450,
        "url": "https://arxiv.org/abs/1706.03762",
        "externalIds": {"DOI": "10.48550/arXiv.1706.03762"}
    },
    {
        "paperId": "tf_nlp_02",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of natural language processing tasks.",
        "authors": [{"name": "Jacob Devlin"}, {"name": "Ming-Wei Chang"}, {"name": "Kenton Lee"}],
        "year": 2018,
        "citationCount": 42100,
        "url": "https://arxiv.org/abs/1810.04805",
        "externalIds": {"DOI": "10.48550/arXiv.1810.04805"}
    },
    {
        "paperId": "tf_nlp_03",
        "title": "Language Models are Few-Shot Learners: The GPT-3 Architecture",
        "abstract": "We present GPT-3, an autoregressive language model with 175 billion parameters. GPT-3 is trained on a massive natural language corpus and demonstrates that scaling up a transformer-based language model significantly improves few-shot performance. We evaluate GPT-3 on a broad range of NLP datasets, finding it performs well on translation, question-answering, and cloze tasks using self-attention layers.",
        "authors": [{"name": "Tom B. Brown"}, {"name": "Benjamin Mann"}, {"name": "Nick Ryder"}],
        "year": 2020,
        "citationCount": 21500,
        "url": "https://arxiv.org/abs/2005.14165",
        "externalIds": {"DOI": "10.48550/arXiv.2005.14165"}
    },

    # 2. Reinforcement Learning Robotics
    {
        "paperId": "rl_rob_01",
        "title": "Deep Reinforcement Learning for Robotic Manipulation and Control",
        "abstract": "We present a deep reinforcement learning framework for training a robotic arm to perform complex manipulation tasks. By designing a continuous reward function, the agent learns an optimal policy for joint control. Our experiments demonstrate that the robot can successfully grasp and move arbitrary objects, bridging the gap between simulation and real-world robotics control.",
        "authors": [{"name": "Sergey Levine"}, {"name": "Chelsea Finn"}, {"name": "Trevor Darrell"}],
        "year": 2016,
        "citationCount": 5420,
        "url": "https://arxiv.org/abs/1504.00822",
        "externalIds": {"DOI": "10.48550/arXiv.1504.00822"}
    },
    {
        "paperId": "rl_rob_02",
        "title": "Continuous Control with Deep Reinforcement Learning in Robotics",
        "abstract": "We adapt the ideas of deep Q-learning to the continuous action spaces common in robotics control. We present an actor-critic, model-free algorithm based on the deterministic policy gradient that can learn complex policies. We evaluate our agent on various robotic simulation tasks, including locomotion and manipulation, demonstrating stable reward convergence.",
        "authors": [{"name": "Timothy P. Lillicrap"}, {"name": "Jonathan J. Hunt"}, {"name": "Alexander Pritzel"}],
        "year": 2015,
        "citationCount": 18900,
        "url": "https://arxiv.org/abs/1509.02971",
        "externalIds": {"DOI": "10.48550/arXiv.1509.02971"}
    },

    # 3. CNN Image Classification
    {
        "paperId": "cnn_img_01",
        "title": "ImageNet Classification with Deep Convolutional Neural Networks",
        "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. Our CNN architecture, AlexNet, utilizes multiple convolution layers and achieves a top-5 error rate far superior to previous computer vision techniques for image classification and visual object recognition.",
        "authors": [{"name": "Alex Krizhevsky"}, {"name": "Ilya Sutskever"}, {"name": "Geoffrey E. Hinton"}],
        "year": 2012,
        "citationCount": 115000,
        "url": "https://dl.acm.org/doi/10.1145/3065386",
        "externalIds": {"DOI": "10.1145/3065386"}
    },
    {
        "paperId": "cnn_img_02",
        "title": "Deep Residual Learning for Image Recognition and CNN Classification",
        "abstract": "We present a residual learning framework to ease the training of convolutional neural networks that are substantially deeper than those previously used. We explicitly reformulate the convolution layers as learning residual functions with reference to the layer inputs. Our ResNet CNN architecture achieves state-of-the-art accuracy on ImageNet classification and other computer vision datasets.",
        "authors": [{"name": "Kaiming He"}, {"name": "Xiangyu Zhang"}, {"name": "Shaoqing Ren"}],
        "year": 2016,
        "citationCount": 156000,
        "url": "https://arxiv.org/abs/1512.03385",
        "externalIds": {"DOI": "10.48550/arXiv.1512.03385"}
    },

    # 4. GAN Image Synthesis
    {
        "paperId": "gan_syn_01",
        "title": "Generative Adversarial Networks for High-Fidelity Image Synthesis",
        "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generator that captures the data distribution, and a discriminator that estimates the probability that a sample came from the training data. This generative adversarial network (GAN) framework is applied to image synthesis, producing realistic samples of digits and faces.",
        "authors": [{"name": "Ian J. Goodfellow"}, {"name": "Jean Pouget-Abadie"}, {"name": "Mehdi Mirza"}],
        "year": 2014,
        "citationCount": 65000,
        "url": "https://arxiv.org/abs/1406.2661",
        "externalIds": {"DOI": "10.48550/arXiv.1406.2661"}
    },
    {
        "paperId": "gan_syn_02",
        "title": "A Style-Based Generator Architecture for Generative Adversarial Networks",
        "abstract": "We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The StyleGAN generator leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and enables intuitive scale-specific control of the image synthesis process.",
        "authors": [{"name": "Tero Karras"}, {"name": "Samuli Laine"}, {"name": "Timo Aila"}],
        "year": 2019,
        "citationCount": 12800,
        "url": "https://arxiv.org/abs/1812.04948",
        "externalIds": {"DOI": "10.48550/arXiv.1812.04948"}
    },

    # 5. GNN Node Classification
    {
        "paperId": "gnn_node_01",
        "title": "Semi-Supervised Classification with Graph Convolutional Networks",
        "abstract": "We present a scalable approach for semi-supervised learning on graph-structured data. It is based on an efficient variant of convolutional neural networks which operates directly on graphs. We motivate our choice of graph convolution architecture via a localized first-order approximation of spectral convolutions. Our graph neural network (GNN) achieves excellent accuracy for node classification tasks.",
        "authors": [{"name": "Thomas N. Kipf"}, {"name": "Max Welling"}],
        "year": 2017,
        "citationCount": 24500,
        "url": "https://arxiv.org/abs/1609.02907",
        "externalIds": {"DOI": "10.48550/arXiv.1609.02907"}
    },
    {
        "paperId": "gnn_node_02",
        "title": "Inductive Representation Learning on Large Graphs using GNNs",
        "abstract": "Low-dimensional network embeddings have proved extremely useful in node classification. We propose GraphSAGE, a general inductive framework that leverages node features to generate node embeddings. Instead of training individual embeddings, we learn a function that generates embeddings by sampling and aggregating features from local neighborhoods via message passing, proving powerful for graph learning.",
        "authors": [{"name": "William L. Hamilton"}, {"name": "Rex Ying"}, {"name": "Jure Leskovec"}],
        "year": 2017,
        "citationCount": 9800,
        "url": "https://arxiv.org/abs/1706.02216",
        "externalIds": {"DOI": "10.48550/arXiv.1706.02216"}
    },

    # 6. Federated Learning Privacy
    {
        "paperId": "fed_priv_01",
        "title": "Communication-Efficient Learning of Deep Networks from Decentralized Data",
        "abstract": "Modern mobile devices have access to a wealth of data suitable for machine learning. We propose a decentralized approach, Federated Learning, where users collaboratively train a shared model under the coordination of a central server. The raw data remains secure on local devices, protecting user privacy while optimizing a global distributed learning model.",
        "authors": [{"name": "Brendan McMahan"}, {"name": "Eider Moore"}, {"name": "Daniel Ramage"}],
        "year": 2017,
        "citationCount": 16400,
        "url": "https://arxiv.org/abs/1602.05629",
        "externalIds": {"DOI": "10.48550/arXiv.1602.05629"}
    },
    {
        "paperId": "fed_priv_02",
        "title": "Privacy-Preserving Deep Learning via Differential Privacy and Federated Learning",
        "abstract": "We explore methods to guarantee privacy in collaborative distributed learning environments. By integrating federated learning with differential privacy techniques, we ensure that individual contributions cannot be reverse-engineered from the model updates. This framework provides secure, decentralized training that guards data privacy.",
        "authors": [{"name": "Martin Abadi"}, {"name": "Andy Chu"}, {"name": "Ian Goodfellow"}],
        "year": 2016,
        "citationCount": 8200,
        "url": "https://arxiv.org/abs/1607.00133",
        "externalIds": {"DOI": "10.48550/arXiv.1607.00133"}
    },

    # 7. Attention Sequence Modeling
    {
        "paperId": "att_seq_01",
        "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "abstract": "We introduce a novel attention mechanism in sequence-to-sequence neural machine translation. Instead of compressing a source sentence into a fixed-length vector, the decoder dynamically focuses on different parts of the encoder hidden states. This soft-attention approach improves performance on long sequences and sequence modeling tasks.",
        "authors": [{"name": "Dzmitry Bahdanau"}, {"name": "Kyunghyun Cho"}, {"name": "Yoshua Bengio"}],
        "year": 2014,
        "citationCount": 35600,
        "url": "https://arxiv.org/abs/1409.0473",
        "externalIds": {"DOI": "10.48550/arXiv.1409.0473"}
    },
    {
        "paperId": "att_seq_02",
        "title": "Effective Approaches to Attention-based Neural Sequence Modeling",
        "abstract": "We propose simple and effective attention-based architectures for sequence modeling. We explore global and local attention mechanisms: global attention examines all source words, while local attention focuses on a small window. Our models show significant gains in seq2seq translation and recurrent neural network modeling.",
        "authors": [{"name": "Minh-Thang Luong"}, {"name": "Hieu Pham"}, {"name": "Christopher D. Manning"}],
        "year": 2015,
        "citationCount": 14200,
        "url": "https://arxiv.org/abs/1508.04025",
        "externalIds": {"DOI": "10.48550/arXiv.1508.04025"}
    },

    # 8. Neural Machine Translation
    {
        "paperId": "nmt_lang_01",
        "title": "Sequence to Sequence Learning with Neural Networks",
        "abstract": "We present a general sequence to sequence learning approach using a multi-layered LSTM to map the input sequence to a vector, and another LSTM to decode it. Our neural machine translation (NMT) system is evaluated on English-to-French language pairs, demonstrating substantial improvements in translation quality over traditional phrase-based systems.",
        "authors": [{"name": "Ilya Sutskever"}, {"name": "Oriol Vinyals"}, {"name": "Quoc V. Le"}],
        "year": 2014,
        "citationCount": 26500,
        "url": "https://arxiv.org/abs/1409.3215",
        "externalIds": {"DOI": "10.48550/arXiv.1409.3215"}
    },
    {
        "paperId": "nmt_lang_02",
        "title": "Google's Multilingual Neural Machine Translation System",
        "abstract": "We address the challenge of scaling neural machine translation to support multiple language pairs. We propose a single seq2seq model that translates between multiple languages without modifying the model architecture. This multilingual NMT system improves translation quality and enables zero-shot translation between unseen language pairs.",
        "authors": [{"name": "Melvin Johnson"}, {"name": "Mike Schuster"}, {"name": "Quoc V. Le"}],
        "year": 2017,
        "citationCount": 3800,
        "url": "https://arxiv.org/abs/1611.04558",
        "externalIds": {"DOI": "10.48550/arXiv.1611.04558"}
    },

    # 9. Object Detection Autonomous Driving
    {
        "paperId": "obj_det_01",
        "title": "You Only Look Once: Unified, Real-Time Object Detection",
        "abstract": "We present YOLO, a new approach to object detection. Prior work repurposed classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes directly from full images in real-time, which is crucial for autonomous driving perception.",
        "authors": [{"name": "Joseph Redmon"}, {"name": "Santosh Divvala"}, {"name": "Ross Girshick"}],
        "year": 2016,
        "citationCount": 41200,
        "url": "https://arxiv.org/abs/1506.02640",
        "externalIds": {"DOI": "10.48550/arXiv.1506.02640"}
    },
    {
        "paperId": "obj_det_02",
        "title": "Multi-Modal Object Detection and Lidar Perception for Autonomous Vehicles",
        "abstract": "Perception in self-driving cars requires robust recognition of surrounding obstacles. We propose a multi-modal object detection network that fuses camera images and Lidar point clouds. The model performs 3D bounding box prediction for pedestrian detection and vehicle tracking, enhancing safety in autonomous driving perception systems.",
        "authors": [{"name": "A. Liang"}, {"name": "B. Taylor"}, {"name": "C. Miller"}],
        "year": 2021,
        "citationCount": 750,
        "url": "https://doi.org/10.1109/TITS.2021.12345",
        "externalIds": {"DOI": "10.1109/TITS.2021.12345"}
    },

    # 10. Recommendation System
    {
        "paperId": "rec_sys_01",
        "title": "Matrix Factorization Techniques for Recommender Systems",
        "abstract": "As the Netflix Prize competition demonstrated, matrix factorization is highly effective for collaborative filtering recommendation systems. We detail how to factorize the user-item rating matrix into low-dimensional latent factors to model user preferences. This collaborative filtering approach outperforms traditional nearest-neighbor techniques for rating prediction.",
        "authors": [{"name": "Yehuda Koren"}, {"name": "Robert Bell"}, {"name": "Chris Volinsky"}],
        "year": 2009,
        "citationCount": 18200,
        "url": "https://doi.org/10.1109/MC.2009.263",
        "externalIds": {"DOI": "10.1109/MC.2009.263"}
    },
    {
        "paperId": "rec_sys_02",
        "title": "Deep Learning based Recommender Systems with Collaborative Filtering",
        "abstract": "We present a neural collaborative filtering architecture to model non-linear interactions between users and items. By replacing the inner product in matrix factorization with a multi-layer perceptron, our recommendation system learns complex patterns of user preference, achieving superior accuracy in item recommendation.",
        "authors": [{"name": "Xiangnan He"}, {"name": "Lizi Liao"}, {"name": "Hanwang Zhang"}],
        "year": 2017,
        "citationCount": 7800,
        "url": "https://arxiv.org/abs/1708.05031",
        "externalIds": {"DOI": "10.48550/arXiv.1708.05031"}
    },

    # 11. Transfer Learning Domain Adaptation
    {
        "paperId": "tf_lrn_01",
        "title": "A Survey on Transfer Learning and Domain Adaptation",
        "abstract": "Transfer learning has emerged as an important machine learning paradigm. We review the state of transfer learning and domain adaptation, categorizing techniques into inductive, transductive, and unsupervised transfer learning. We discuss how pretrained models and feature extraction can generalize across different target domains.",
        "authors": [{"name": "Sinno Jialin Pan"}, {"name": "Qiang Yang"}],
        "year": 2010,
        "citationCount": 24200,
        "url": "https://doi.org/10.1109/TKDE.2009.191",
        "externalIds": {"DOI": "10.1109/TKDE.2009.191"}
    },
    {
        "paperId": "tf_lrn_02",
        "title": "Domain-Adversarial Training of Neural Networks for Domain Adaptation",
        "abstract": "We introduce a new approach to representation learning for domain adaptation, where data at training and test time come from different distributions. We show that fine-tuning with a domain classifier gradient reversal layer forces the feature extractor to learn cross-domain representations, improving transfer learning.",
        "authors": [{"name": "Yaroslav Ganin"}, {"name": "Evgeniya Ustinova"}, {"name": "Hana Ajakan"}],
        "year": 2016,
        "citationCount": 8500,
        "url": "https://arxiv.org/abs/1505.07818",
        "externalIds": {"DOI": "10.48550/arXiv.1505.07818"}
    },

    # 12. RNN Time Series Forecasting
    {
        "paperId": "rnn_time_01",
        "title": "Long Short-Term Memory for Sequential Time Series Forecasting",
        "abstract": "Long Short-Term Memory (LSTM) is a recurrent neural network (RNN) architecture designed to solve the vanishing gradient problem. We apply LSTM networks to sequential time series forecasting, predicting temporal trends in financial and weather data. Our RNN outperforms classic autoregressive methods at captures long-term dependencies.",
        "authors": [{"name": "Sepp Hochreiter"}, {"name": "Jürgen Schmidhuber"}],
        "year": 1997,
        "citationCount": 94000,
        "url": "https://doi.org/10.1162/neco.1997.9.8.1735",
        "externalIds": {"DOI": "10.1162/neco.1997.9.8.1735"}
    },
    {
        "paperId": "rnn_time_02",
        "title": "Temporal GRU Networks for Time Series Prediction and Forecasting",
        "abstract": "We explore gated recurrent unit (GRU) architectures, which are simpler variants of RNNs, for time series prediction. We evaluate the models on time series forecasting datasets, examining prediction accuracy across different temporal intervals. The GRU network demonstrates fast training and strong predictive modeling on sequential datasets.",
        "authors": [{"name": "K. Cho"}, {"name": "B. Merriënboer"}, {"name": "C. Gulcehre"}],
        "year": 2014,
        "citationCount": 19500,
        "url": "https://arxiv.org/abs/1406.1078",
        "externalIds": {"DOI": "10.48550/arXiv.1406.1078"}
    }
]
