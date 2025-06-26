# A Unified Framework for Macro and Micro Expression Recognition Using Dynamic Facial Graph Modeling and Visual Stream

## Abstract
Macro-expressions (MaEs) or micro-expressions (MEs) exhibit substantial differences in both duration and intensity, often requiring separate frameworks to capture their unique characteristics accurately. Consequently, current facial expression recognition models typically target either MaEs or MEs to enhance human-computer interaction. However, real-world scenarios frequently involve both types of expressions occurring in close succession, making a unified model for macro and micro expression recognition essential for accurately interpreting human emotions. Despite this importance, unified models remain limited, and often struggle with inadequate feature extraction and ineffective integration of spatial-temporal features. To address these limitations, this paper introduces a unified framework combining a facial graph stream (FGS) with a visual stream (VS). First, we propose a landmark motion-based keyframe selection algorithm to address time discrepancies between MaEs and MEs, and a spatial-temporal local directional number (STLDN) descriptor to encode facial dynamics effectively. Subsequently, the FGS employs a graph structure autoencoder to dynamically generate graph structures and incorporates a spatial-temporal landmark attention module to enhance spatial and temporal feature integration. Concurrently, the VS utilizes vision transformer and bi-directional long short-term memory networks to learn spatial and temporal features from the STLDN sequence. Finally, an attentive feature fusion module combines the features from each stream. Experimental results show that our framework significantly improves recognition accuracy for both MaEs and MEs and outperforms state-of-the-art methods. 
 

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