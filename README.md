# A Unified Framework for Macro and Micro Expression Recognition Using Dynamic Facial Graph Modeling and Visual Stream

## Abstract
Macro-expressions (MaEs) or micro-expressions (MEs) exhibit substantial differences in both duration and intensity, often requiring separate frameworks to capture their unique characteristics accurately. Consequently, current facial expression recognition models typically target either MaEs or MEs to enhance human-computer interaction. However, real-world scenarios frequently involve both types of expressions occurring in close succession, making a unified model for MaEs and MEs recognition essential for accurately interpreting human emotions. Despite this importance, unified models remain limited and often struggle with inadequate feature extraction and ineffective integration of spatio-temporal features. To address these limitations, this paper introduces a unified framework combining a Facial Graph Stream (FGS) with a Visual Stream (VS). First, we propose a Landmark Motion-based Keyframe Selection algorithm to address time discrepancies between MaEs and MEs, and a Spatio-Temporal Local Directional Number (STLDN) descriptor to encode facial dynamics effectively. Subsequently, the FGS introduces a novel Spatio-Temporal Graph Autoencoder and Spatio-Temporal Facial Graph Network to dynamically generate graph structures and capture facial appearance as well as motion features from landmarks. Concurrently, the VS utilizes a Vision Transformer and Bidirectional Long Short-Term Memory networks to learn additional auxiliary spatial and temporal features from the STLDN sequence. Finally, an Attentive Feature Fusion module combines the features from each stream. Experimental results show that our framework significantly improves recognition accuracy for both MaEs and MEs as well as outperforms state-of-the-art methods. 
 

## Main Requirements
This code package was developed and tested with Python 3.9.

* numpy==1.25.2
* opencv==4.8
* pandas==1.3.2
* scikit-learn 0.24.2
* scipy==1.11.1
* torch==2.2.2


## Dataset
* [AFEW] (https://users.cecs.anu.edu.au/~few_group/AFEW.html)
* [Oulu-CASIA] (https://www.oulu.fi/en/university/faculties-and-units/faculty-information-technology-and-electrical-engineering/center-for-machine-vision-and-signal-analysis)
* [CAS(ME)<sup>2</sup>] (http://casme.psych.ac.cn/casme/e3)
* [CAS(ME)<sup>3</sup>] (http://casme.psych.ac.cn/casme/e4)

## Running the code

Install the required dependency packages

  ```
python main.py --csv_path --image_root --catego --num_classes --batch_size --epochs 
  ```

## License

A Unified Framework for Macro and Micro Expression Recognition Using Dynamic Facial Graph Modeling and Visual Stream is released under the MIT license.
