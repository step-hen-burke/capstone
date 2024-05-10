# Deep Learning Model Compression for Resource-Constrained Environments

This proposal pertains to a capstone project under the topic of investigating and implementing compression techniques to enable the use of deep learning models under resource-constraints in order to broaden the applicability of deep learning models, enable use-cases under strict constraints which would not be possible without compression, and save costs associated with storage, inference time, and energy usage in existing applications.

The domain area of the proposed research is Machine Learning (deep learning in particular) compression, and the proposed problem area is resource constrained model deployment.

Deep learning models are widely applicable in many domains, but can require significant resources to store and perform inference with, which may limit their use in certain environments such as mobile devices and IoT devices. 

This project seeks to explore the constraints and resource considerations that deep learning models are subject to, such as limited memory, processing power, energy consumption, cloud storage and retrieval costs, model artifact I/O time and bandwidth usage, etc.

The project also seeks to experiment with how these constraints can be alleviated in order to save costs associated with machine learning inference and broaden the scenarios in which deep learning can be applied.

## Research Objectives

The proposed capstone project has been organised under 5 research objectives, these are structured in such a way that the project can effectively identify state of the art deep learning compression techniques, discover scenarios in which these techniques are typically applied according to industry best practices, evaluate and experiment with the techniques in order to ascertain  Full details of the objectives are as follows:

### Research Objective 1

Identify and compare current approaches to deep neural network compression, taking note of the impact of different network architectures and data structures on the applicability and effectiveness of different approaches. Classify approaches by the overarching technique, and based on the resource constraints they can help address: inference speed, memory overhead, and energy usage are commonly seen metrics within the literature.
### Research Objective 2

Identify typical scenarios within which a deep neural network would be subject to resource constraints, acceptable performance/resource trade-offs, and typical speed and footprint benchmarks that a compressed model should be evaluated against (through interview with industry experts, for example). Additionally, identify and experiment with practical use-cases which are enabled or enhanced only by the compression of deployed models.

### Research Objective 3

Using standard datasets (MNIST, CIFAR, Treebank, for example. Or others as deemed relevant) and architectures, evaluate the effectiveness of compression techniques in terms of: disk size reduction, memory usage, predictive performance preserved, inference speed, etc. - particularly with regard to important metrics and consumer hardware benchmarks such as typical mobile or IoT hardware.

### Research Objective 4

Develop a suite of tools to compress predictive models for use in resource-constrained environments using combinations of techniques which are deemed to be optimal. Optimal in this context would involve trade-offs between factors explored in objective 3 - model size, inference speed, predictive accuracy, etc. Identify viable compression strategies that utilise combinations of previously explored techniques. The use of hybrid compression techniques has been identified by multiple authors in the literature as an area of interest for further work.

### Research Objective 5

Demonstrate the applicability, advantages, trade-offs, etc. of compressed models with regard to specific hardware configurations and typical resource constraints as explored in objectives 2 and 3, using techniques and learnings found as a result of objective 4.


## Primary Research 

### In-depth Interviews to motivate experimental scenarios

The project proposes to undertake primary research in the form of in-depth interviews as part of research objective 2. These interviews will support the applicability of model compression techniques by identifying scenarios where models are used in industry under resource constraints, discovering which constraints are most relevant to businesses and practitioners, and obtaining anecdotes about applications that were made possible or otherwise enhanced by model compression techniques and/or by overcoming resource constraints.

As the use of deep learning models is relatively niche within industry, and the use of these under strict resource constraints even more so, the use of non-probability sampling methods is justified - particularly as this phase of primary research is concerned with contextualising and motivating further work in a qualitative manner. The proposed sampling strategy for this primary research is a mixture of convenience leading into snowball strategy. The author has nearly a decade of experience working within the data analytics field, and intends to reach out to current and former colleagues who are known to the author to have either worked or currently work with deploying models subject to resource constraints. It is hoped that the initial cohort of interviewees will be able to connect the author with other relevant persons within their professional networks. Relevance in this sense will be adapted as the needs of the project unfold - for example, a particular use-case may be discussed in an interview which would then warrant a further interview with a person who worked more closely on this use-case. There is potential for bias in this approach in that the sample is largely limited to the extended professional networks of the author and interviewees, and it is possible that multiple interviewees may work or have worked in the same company for example.

The author notes that there is an additional potential for bias in this area, in that the interviewees will not be aware of all model compression techniques, and indeed may not be aware of any, such that the interviewees personal experience may not be indicative of true best practices. This may be compounded by the aforementioned limitations of snowball sampling, in that those within the same professional network may be aware of certain techniques from know-how gained from each other or a common company or peer. However, it is hoped that gaining knowledge of real-world industry experience and anecdotes of use-cases in this field will help to motivate scenarios to be tested as part of the experimental research objectives that follow on from this.

Additionally, the author notes the importance of keeping interview responses non-identifying where possible. As the interviewees accounts of particular scenarios will most likely pertain to their professional work, it is important to be aware that discussion of their work may touch on trade-secrets or industry know-how, and it is an ethical consideration to protect these where possible on behalf of the interviewee so as to not impact their relationship with their employers.

### Experimentation 

The proposed project intends to perform experimentation in several areas of deep learning compression as its primary focus in addressing research objectives 3, 4, and 5. It has been identified in the literature that there has been limited work into the composition of deep learning compression techniques, and examples of applications involving more than one technique have not explored the optimal combination, order of application, etc. The author intends to perform experimentation with respect to this. As part of research objective 3, experiments are to be carried out to measure how the application of model compression addresses resource-constraints such as inference speed, and memory footprint. As is common in the literature, these experiments would be carried out on publicly available, pre-trained models where possible. This both alleviates the computational power and time investment needed to train larger models from scratch, enabling the application of compression techniques to models that might be infeasible to train on hardware available to the author, and allows for any obtained results to be more easily reproduced as network compression would take place on standardised models. 

The author notes that there are ethical and potentially legal considerations here; pre-trained models are commonly used for benchmarking within the literature surrounding deep learning compression, but it should be noted that the use of pre-trained models means that additional care should be taken around the sourcing of trusted models and the veracity of the data used to train them. It should also be ensured that it is permissible within the licenses of all obtained models to modify them via compression. It is also of note however that the use of pre-trained models for experimentation alleviates some ethical issues surrounding the handling and processing of potentially sensitive or proprietary information used to train said models in the first place.

## Literature Review

This literature review seeks to motivate the use of neural network compression in order to increase the applicability of deep learning within resource constrained environments and as such has been organised thematically. 

It first gives an overview of commonly seen resource constraints such as those found in edge devices or internet of things (IoT), and explores deep learning applications that might be commonly found within these constrained environments. Additionally, the typical hardware within these domains is explored.

Following this is an overview of common neural network compression techniques, categorised into five identified fields based on the overarching approach, and drawing on both seminal and state of the art examples. For each of the these, how the technique addresses specific resource constraints can differ, and indeed which constraints are addressed at all can differ.  

Furthermore, an overview of common benchmarking strategies in the field of deep learning compression is provided. This explores common datasets and models used to benchmark compression techniques, discusses metrics commonly used in order to quantify a techniques efficacy in addressing a specific resource constraint,  discusses the applicability of technique categories to specific models, metrics, and use-cases, and identifies practical trade-offs that may be made when choosing to use compression techniques and selecting between them.

Finally, an overview of potential further work and commonly seen areas of interest within the literature will be provided.

### Resource Constraints in Machine Learning

Profentzas et al. (2021) describe three key constraints that can be addressed by neural network compression; memory, energy, and computational complexity. They note that models typically require megabytes or more of storage space and RAM, whereas a resource constrained device may only have kilobytes available. The computational complexity of inference limits the applicability of models, as a resource constrained device may consume excessive power and take excessive time in order to perform inferences, thus precluding its use in power constrained or time sensitive applications. Additionally, a typical strategy to alleviate computational complexity - hardware accelerators - may not be available on a resource-constrained device or may be economically infeasible to use. As such, strategies to reduce the resource overhead and number of operations required for inference via compression are preferable.
#### Deep learning within the Internet of Things (IoT)

Although resource constraints can be valid concerns at all computational scales, the challenges and limitations become particularly pertinent when dealing with applications within the Internet of Things (IoT).

IoT applications are comprised of networks of small, low energy units containing one or more sensors and limited computational resources. Despite the "smallness" of their individual components, IoT applications typically have all of the properties of big data systems, satisfying all 5 "Vs" - Volume, Velocity, Variety, Veracity, and Value, with multiple devices capturing a large amount of real-time or near real-time data from potentially heterogeneous sensors which needs to be transformed in order to serve a purpose to the system as a whole. 

Mohammadi et al. (2018) identify many machine learning applications within the IoT space - all of these are subject to resource constraints and can benefit from lowering the overheads of their deployment. Reducing the memory footprint of models would allow for more complex models to fit on the limited memory available in IoT devices. Reducing power consumption would reduce costs to the end-user, both in terms of reducing their power bill for hard-wired devices, and minimising the frequency with which a user has to charge or replace the batteries of standalone devices. Reducing inference time would increase the efficiency of systems as a whole, mitigating or removing computational bottlenecks and enabling real-time applications on edge devices which were previously infeasible.

Profentzas et al. (2021) argue that performing inference on edge devices rather than on a centralised monolithic processor has several advantages; each device processing its own sensor data and sending a transformed aggregate can reduce network overhead compared to sending the raw telemetry, thus increasing the efficiency and speed of the distributed application, additionally, as each device has sole custody of its raw data, there is no single point of failure with which an attacker can gain access to all of the potentially sensitive or private information held by the network, it also mitigates the potential attack vector of intercepting private data in-transit around the network, as systems can be designed such that this data never leaves the edge device without being transformed. Both of these advantages to performing inference on edge devices are particularly pertinent to IoT applications, both from a resource and security perspective.

Within IoT literature, particular concern towards data privacy and security is commonly seen. IoT devices have previously been targeted in attacks due to typically lax security standards from IoT device manufacturers (Fernandez et al., 2016), their ubiquity, and the nature of many such devices being "set it and forget it", in that a user is unlikely to notice that a headless sensor has been compromised compared to their laptop or PC on which they may notice slowdowns or periodically run anti-virus software. One such high profile case was the Mirai botnet, which in 2016 performed a series of distributed denal of service attacks against high profile targets such as Amazon and Netflix using a network of over 600 thousand compromised IoT devices (Antonakakis et al., 2017).

Selvan et al. (2023) discuss resource constrained machine learning within a health context. They posit that increasing the ease of deployment of machine learning models can in turn increase access to machine learning enabled diagnostic tools which would otherwise require specialised hardware and/or a significant computational investment. 

Typical ML applications using model compression include image recognition, and speech recognition. These are natural targets for compression as they are useful when deployed on resource constrained edge devices (IoT).

Mohammadi et al. (2018) identify several image and speech recognition applications that utilise IoT networks and deep learning. They also note that network compression is a viable approach to adopting deep networks to resource constrained devices, in addition to other possible tools and techniques. such as approximate computing and specific hardware accelerators. Some of the use cases they identify include food recognition, traffic sign detection, and road damage detection within the field of image recognition, and text to speech applications within the field of audio and language processing. 

Baller et al. (2021) compared several IoT devices for image classification tasks, the ASUS Tinker Edge R, Raspberry Pi 4, NVIDIA Jetson Nano, the Google Coral Dev Board, and the Arduino Nano 33 BLE. They noted that these devices typically had low amounts of available RAM (a choice of 1Gb or 4Gb in the case of the Coral Dev, but only 256kb in the case of the Arduino Nano), and relatively limited processors. However they also note that the performance of certain edge devices can be enhanced by dedicated hardware acceleration, as was the case with NVIDIA Jetson Nano which has a GPU.

Lin et al. (2018) discuss speech recognition in the context of mobile hardware (EdgeSpeechNets), and were able to train alternatives to the state of the art res15 model using the Google speech commands dataset. Using tensorflow mobile their model was able to run on a Motorola phone with a 34ms latency (10x faster than res15) and 1mb footprint (16.5x smaller than res15).

Although the above section has primarily dealt with resource constraints at the small scale, it is important to note that these constraints exist at large scales also. The promising results and subsequent growth in both size and popularity of modern LLMs has made it so that these models are quite often on a scale where they can no longer fit on a single GPU/TPU as noted by Bai et al. (2024). Additionally, as more demands are being placed on these models, their performance and cost in terms of inference speed, compute cycles, and memory footprint become an important business consideration for those companies who rely on them. 

### Neural Network Compression

#### Weight Sharing

Weight sharing is the process of having multiple weights in a neural network be shared across layers rather than storing one value for parameter. Mishra et al. (2020) note that weight sharing techniques can be effective at reducing both the storage and computation requirements of a neural network. One classical example of weight sharing was proposed by Nowlan and Hinton (1992) in which a Gaussian mixture model was used to assign weights to groups which would share a value.

It is notable that weight sharing is typically applied prior to training as opposed to applying a compression technique on a previously trained model, however some techniques such as those described by Chen et al. (2015) are applicable post-training.
#### Quantization

Quantization aims to reduce the size of the numerical representations of parameters within a model. A model trained with 32 bit numbers may be found to have comparable accuracy with 16 bit, or even 8 bit representations. Integer representations may also be used rather than the typical 32 bit floating point representation. Taken to the extreme, binary networks can represent each weight as a binary number. The change in representation can lead to drops in performance, particularly with more extreme quantization processes, however it has been shown (Gysel et al. 2018) that retraining quantized networks can help mitigate these performance losses.

Some quantization techniques use mixed precision networks, in which some parameter values (those particularly sensitive to quantization and likely to lead to performance loss) are kept fully represented, and those less likely to lead to performance loss are represented at lower levels of precision. Micikevicius et al. (2017) used a mix of full and half precision parameters in order to significantly reduce model memory overhead without comprimising accuracy. Modern techniques such as iterative product quantization (Stock et al., 2019) and hessian based quantization (Dong et al., 2019) involve selective mixed precision quantization, with quantization degrees differing layer by layer in order to minimise performance loss.

#### Pruning

Pruning seeks to cut a model down into its most performant sub-network by reducing the number of parameters contained in a pre-trained network. O'Neill (2020) notes that pruning may degrade network performance and as such may require additional retraining steps post-pruning, although whether this is necessary depends on the technique in question.

One classical pruning technique proposed by Moser and Smolensky (1989) is skeletonization. This seeks to estimate the least important model units during training and remove them, thus lending the technique its name, as only the minimal important structure within the model is maintained. This technique assigns an importance to each weight within the network and iteratively removes them. However, O'Neill et al. note that this, and other, methods of removing weights one at a time is infeasible for modern networks as the number of parameters is many orders of magnitude higher than when these techniques were originally designed.

Optimal Brain Damage (Le Cun et al., 1990), and a further improvement Optimal Brain Surgeon (Hassibi and Stork., 1994) estimate weight importance in such a way that many of the importances are shared between weights, and as such, portions of weights are pruned at each iteration rather than one at a time.

Cheng et al. (2020) chose to group Quantization and Pruning together in their taxonomy of model compression techniques as they note that both methods fundamentally aim to eliminate redundancy within the models. However, it is the author's belief that these two categories are distinct in their approach and warrant being separated into separate compression technique families. Cheng also notes that an inherent disadvantage of pruning is that techniques which use L1 or L2 regularization typically require more iterations to converge, and although the pruning family of compression techniques are usually effective at reducing model size, they do not address model efficiency, in that inference speed is unlikely to improve.

#### Low Rank Approximations

Low rank approximations seek to decompose networks in such a way that their components can be expressed as a lower dimensional representation.

Cheng et al. (2020) note that low rank approximation methods are typically computationally expensive as they are typically layer-by-layer rather than globally applied.

Tensor Decomposition

#### Knowledge Distillation

Knowledge distillation methods aim to use the learnings from large, highly parameterised models (teacher networks) to train small, performant  models (student networks). Bucila et al. (2006) successfully trained a small student network using labels provided by a much larger teacher network. They note that the original complex ensemble model was sufficiently large and slow that it would be infeasible to use in many situations, however their compressed network was 1000 times smaller and 1000 times faster with a negligible loss in performance.

Lopes et al. (2017) showed that it is possible to perform knowledge distillation compression even in scenarios where the original data is not accessible, which may be the case due to privacy concerns, issues of ownership, or other issues rendering the original data inaccessible.

O'Neill et al. (2020) note that knowledge distillation can be applied to transformer networks - BERT (Devlin et al, 2018) being the most prominent at the time, however it stands to reason that similar work could be carried out on modern networks such as GPTs. 

Despite wide applicability of the technique, Cheng et al. (2020) noted that knowledge distillation works best in problem areas where the source dataset is relatively small, and particularly when significant efficiency improvements are required. 

---

It it notable that although the taxonomy of compression techniques presented here has delineated between groups of techniques by the specifics of their implementation and overall goal of their approach, it is also possible to group these at a higher level by categorising them by whether or not the technique is supervised or unsupervised. For example, low rank approximation and decomposition techniques are typically unsupervised, whereas knowledge transfer based techniques are typically supervised.
### Existing Frameworks, Benchmarks, and Architectures

Certain frameworks exist in order to compress existing model artifacts into more efficient forms, and to target certain architectures. Profentzas et al. (2021) compared three different IoT focused deep learning frameworks; tensorflow-lite-micro, uTensor, and CMSIS-NN using the MNIST and CIFAR-10 datasets. They found that while each framework gave performance and efficiency gains, there was a trade-off between ease of use and usefulness, in that the most efficient final models required significant customisation in C in order to get the most out of the compression framework. It is also notable that this study focused exclusively on quantization as a compression method, and focused on CNNs as the model architecture for compression as other network architectures had limited support within the chosen frameworks.

Cheng et al. (2020) note that AlexNet, LeNets, and All-CNN-nets are the most well known models in compression assessment literature, however, it may be the case that the general body of research was geared towards CNNs and image applications at the time rather than transformers and text applications that are now common post-Chat-GPT. The authors acknowledge that there was some work that had been carried out for tasks beyond image classification at the time. Zhu et al. (2023) discussed model compression within the context of language models, and noted that many of the techniques discussed above (knowledge distillation, quantization, pruning, etc.) are applicable to language models and are relevant within an NLP context. Tang et al. (2024) discuss neural network compression for transformer models in detail, and also found that many of the previously discussed techniques are applicable, although they note that the different architecture necessitates special considerations when applying compression techniques to transformers as opposed to CNNs or RNNs for example. 

Baller et al. (2021) described benchmarking models on edge devices, in addition to describing special considerations when targeting device hardware. They chose power consumption, inference time, and model accuracy as their key metrics across multiple device types, and chose to evaluate models according to the number of devices they could be optimised for. In this case, it is possible that certain models might have been suitable to be optimised for a certain device, but were not considered due to incompatibility with other devices. They also noted that even very long (11+ second) inference times may be acceptable in real world applications when balanced with advantages in power consumption. It is worth noting that optimisations gained via neural network compression are advantageous within their application context, and trade-offs can and should be made within bounds that are deemed acceptable for the task at hand. Also of note was that the raspberry pi, a relatively cheap and commonly available single board computer performed competitively across all evaluations despite a lack of dedicated hardware acceleration.

### Discussion and Further Work

O'Neill et al. (2020) notes that while extensive research and experimentation has taken place with deep learning compression techniques in isolation, comparatively little work has gone into discovering optimal combinations of techniques. Within the examples of hybrid compression applications identified by the O'Neill, the optimal order in which to apply the compression techniques isn't dealt with, and this is noted as an area of future work. Ishtiaq et al (2021) do address this, but their work was limited to CNNs and only dealt with compression via pruning and quantization. A recent paper (Shen et al. 2024) also investigated chaining multiple compression techniques, including knowledge distillation in addition to pruning and quantization, however their focus was also exclusively CNNs, it is possible that similar approaches could be valuable if applied to other network architectures.

As noted above, most of the literature has dealt with image and speech applications, as these were the most promising areas of exploration at the time of writing. Many modern applications of machine learning deal with text data and natural language processing given the success of transformers, LLMs, and the popularisation of models such as Chat-GPT. Bai et al. (2024) discuss the applicability of compression techinques such as pruning, quantization, and knowledge distillation to modern LLMs. O'Neill et al. (2020) noted at the time that model compression techniques had seen a resurgence with the advent of deep convolutional and transformer architectures, so as new and larger models become available it seems reasonable that the application of both traditional and state of the art compression techniques to these modern models should be addressed and explored. Zhu et al. (2023) echo this sentiment, noting that although there have been attempts at employing hybrid compression strategies within the context of language models, no systematic study on combining compression techniques had been carried out. This was also identified as a promising direction for exploration by Tang et al. (2024) in the context of transformer models.

As discussed previously, resource constraints exist at all scales of deep learning application, and the advantages of creating a performant compressed model should not be eschewed in favour of addressing resource constraints with additional hardware. As noted by Bai et al. (2024), it is in the nature of modern machine learning models to grow exponentially in scale such that they require more resources than can be provided even at great expense.

### References

