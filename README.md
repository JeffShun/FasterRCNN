# This is a personal implementation of FasterRCNN based on PyTorch
- Backbone: ResNet50+FPN
- Only the last but two layer is selected for regression and classification in this project for small object detection
- Other settings are basically consistent with those of the official version

# Demo training and testing
1. Prepare the original training data
- Create a new train_data/origin_data folder under the train folder and put the original training data into it. Make sure json files of label are derived from labelme!
2. Generate the processed training data, which will be stored in the train_data/processed_data folder
- Run **cd train**
- Run **python custom/utils/generate_dataset.py**
3. Start training
- Run **python train.py**
4. Prepare the test data
- Put the prediction data into the test/data/input directory.
5. Start prediction
- Run **cd test**
- Run **python main.py**
6. Evaluate the results
- Run **python analysis_tools/eval_matrics.py**
