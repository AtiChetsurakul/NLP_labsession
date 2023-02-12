# NLP_labsession

--------------------
## Course From AIT
-------------------
## Weekly paper reading
| Paper            | main_point                                                                            | - Problem                            | - important of problem                                                                                                     | - Prev work                           | - New approch                                                                                                                                                                                                                                                                                                                | - Justification                                           | - what do we learn?                                                        |
|------------------|---------------------------------------------------------------------------------------|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------|
| [0] Pixie, Haque | With Pixie, new dataset,  implicit and indirect classification performance are boots. | implicit and indirect classification | To filter unrelate stuff/comment. - For Example, shopping online review since some people  tend to make a indirect review. | CompSent-19 can make a 87.4% F1 score | Dataset including 1 Comparative (indirect comparison)  2 Implicit ( one compared entities are mention) 3 Explicit (mentioned both) - need abbreviations, and [inter-rate agreeM](https://en.wikipedia.org/wiki/Inter-rater_reliability) - Dataset can train with traditional ML tranformer based(TF) and TF with segment emb | - Out perform Model from Compsent dataset by 4%  to 6.3%  | - Incase we need to do some project we might need to make our own dataset. |

<br>

## Week 2
<br>


| Paper         | Liang , JointCL                                                                                                                                                                                                                                                                                                                                                                                         |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| main point    | With JointCL framework, the infernce model reach the state-of-the-art performance on ZSSD task.                                                                                                                                                                                                                                                                                                         |
| Problem       | stance information of an unseen target can be in the knows target from ` target-aware perspective`                                                                                                                                                                                                                                                                                                      |
| Prev work     | Stance detection Aims to identify people's standpoint and attitude towards a target. <br> Zeroshot stance detection -> model to detect stance towards unseen targets                                                                                                                                                                                                                                    |
| New approch   | JointCL framework can leverage the stance features of known targets from both context-aware and target aware perspective <br> Stance Contrastive Learning -> imporve quality of stance feartures by leverae the similarity if traning instance <br> Target-Aware Protitypical Graph Contrative Learning, a novel one, use for learn the correlation and difference among the target-base representation |
| Justification | out perform state-of-the-art in ZSSD task in 3 datasets                                                                                                                                                                                                                                                                                                                                                 |
| How to use    | - Not yet I need to look in prev work first - from `Adversarial Learning for Zero-Shot Stance Detection on Social Media`                                                                                                                                                                                                                                                                                |
|               |                 




-----------------------
## Homework paper reading


-------------------------

## reference
    - [0] Amanul Haque, Vaibhav Garg, Hui Guo, and Munindar Singh. 2022. Pixie: Preference in Implicit and Explicit Comparisons. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 106–112, Dublin, Ireland. Association for Computational Linguistics.

    - [1] Bin Liang, Qinglin Zhu, Xiang Li, Min Yang, Lin Gui, Yulan He, and Ruifeng Xu. 2022. JointCL: A Joint Contrastive Learning Framework for Zero-Shot Stance Detection. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 81–91, Dublin, Ireland. Association for Computational Linguistics.