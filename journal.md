
# AI-TrashCan
#### By Kush Desai & Heyzac Yong


### Journal

This is our Journal that we have been writing since Jan 10, 2026. I really hope achieves its goal of letting you see how we were thinking throughout this project. So let's get into the project. You can also find our project's report on [https://tinyurl.com/basef-report](https://tinyurl.com/basef-report)

## Jan 10, 2026
**Time: 3:30pm**

**What we did today:**

Me and Heyzac were talking on voice call possible ideas that we could follow for our project. We considered making out project about 3D printers whether that be making a custom one or one with good error detection. We came up with the idea of a trash classifier. We really liked the idea as we often saw people in class confused which bin the item they are trying to throw away goes in.

## Jan 11, 2026
**Time: 4:50pm**

**What we did today:**

Heyzac did some research about the project such as how bin the problem itself was. He discovered that, according to the [EPA](https://www.epa.gov/facts-and-figures-about-materials-waste-and-recycling/national-overview-facts-and-figures-materials#NationalPicture), almost 25% of trash in the recycling bin is not supposed to be there. This confirmed that misclassification is a big issue so we started thinking about how it would work. We knew that a important part of this project would be to make it autonomous so I also did some research about how we could make it autonomous. In the end we both agreed to use AI as the center of our project as that would be the best way to make it autonomous.

## Jan 17, 2026
**Time: 2:25pm**

**What we did today:**

Today we were talking about how we want to start building the project itself and we both agreed that we should start by making the code and then work on the body of the trashcan. We started by doing some research about image recognition models. We wanted it to work locally if possible so we looked for a lighweight model. We happend upon google's Mobile net V2 and we decided to use that. We were also wondering how we would train it so we read some articles about it and we landed on the python libraries torch and torchvision.

## Jan 18, 2026
**Time: 1:15pm**

**What we did today:**

We decided to test out the model today to see if it can predict what the trash is. We gave it an image of an apple core and it said that there was an 15% chance that it was an apple. We realized this model was most likely trained on day-to-day items that were not trash. As such we explored some alternative modles but came up empty. We decided did some research about how to fix this problem and we found out that you can removed the final classification layer and retrain it to classify again. We moved forward with this idea and decided to train it to recognize materials instead of items. This would mean that the image of the apple core would return organic instead of apple. We thought this was the smarter approach as it would remove the neccesity to train it on all possible items and we would just have to train it on each material instea. We finilazed the code to train and realized that we did not have the neccasary dataset to train it. We looked online and very happily found waste-garbage-management-dataset by omasteam on huggingface which is a library of AI-models and datasets. We tried downloading it via the huggingface cli but it told us we had made to many requests as the dataset had 19.8k images in it. So we planned to try downloading the dataset the next time we worked on the project.

## Jan 24, 2026
**Time: 4:40pm**

**What we did today:**

We came back and tried to download the rest of the dataset but it did not work as it treats each image as a request and if we gave their server 19.8k requests in a short amount of time they would think we are trying to give them a [DoS](https://www.cloudflare.com/learning/ddos/glossary/denial-of-service/) attack. We looked at different ways to isntall the dataset and landed on [GIT LFS](https://git-lfs.com/) which allows us to download it via [GIT](https://git-scm.com/) instead. We ran into some difficulties setting up GIT LFS but we downloaded it relatively quickly. We knew that training an AI model would use a lot of cpu on my computer so we decided to let it run overnight so it would not have any inturuptions. Just as another precaution we ran the train file with nohup and caffeinate in the background which just tells the computer to stay up all night and make sure this file keeps running.

## Jan 25, 2026
**Time: 1:45pm**

**What we did today:**

We were really excited today so we opened the computer early in the morning to check if it was done and it sadly wasn't. Interestingly when we opened activity manager to see how the cpu was doing it was and we noted that next to the python program it said it was using 300% cpu (this means it was using 3 of the 10 cores) and there was around 75% cpu idle (over all the cores) which showed that the macbook air m4 chip had no issues running through the night. When it was finshed we were shocked by the results. We had run a total of 10 epochs and in the end we had a valitdation accuracy of 95.4%. Below I have the accuracy related to each material that can help us know which materials it has problems with.
```
Per-Material Accuracy:
  plastic   :  89.4%
  paper     :  94.6%
  metal     :  89.7%
  glass     :  93.3%
  organic   :  97.5%
  other     :  98.0%
```
The confusion matrix also told us that it had a problem mainly with plastic and glass as it confused one for the other multiple times
```
Confusion Matrix:
(Rows = actual material, Columns = predicted material)
[[ 361    7    4   23    0    9]
 [   5  672    5    5    3   20]
 [   1    3  175    8    0    8]
 [  15    2   10  586    2   13]
 [   0    2    0    1  199    2]
 [   7   15    9    2    4 1775]]
```

We edited the file we were using earlier to use the new model we had created and it had much better accuracy. We tested it with three files:
1 image of an apple core 
1 image of a plastic bottle
1 image of a piece of paper

The result was a 100 confidence that the plastic bottle was made of plastic and the pape was made of paper and a 80% confidence that apple was an organic.

The apple was confused with paper probably due to the white background or just the fact that organics often have unique features while plastic and paper have definitve properties.

I would also like to mention here that this was just a test to make sure it works and that we plan on doing a more rigourous test int the future.

## Jan 31, 2026
**Time: 5:00**

**What we did today:**

Today we started with the goal to make a server that accepts HTTP requests with images and returns the result. I had previous experience with FASTAPI servers so we decided to make one. After installing the libraries and writing a simple GET endpoint on "/" that returns {"Hello":"World"}, we started writing the "/classify" endpoint. We knew that we did't need to rewerite the entire classification code so we just imported the main classifiying function and passed it the image recived from the endpoint. We used the auto-generated swagger document to test it. We passed it the same three images from eariler but we ran into an issue. It turned out that when the image arrived from server via the endpoint it was a PIL object and this was differfent from what we were doing earler where we were giving it an .png image. So we had to rewrite the function to accept pil objects and we imported and called that function instead. We tested it again and it was working giving us the same results as the earlier tests. With this we realized that we had finished the backend code of our project and we just had to do everything regarding the actual body, such as the code and the wiring of the sensors.

## Feb 01, 2026
**Time: 3:10**

**What we did today:**

Today we worked on the two motors that we plan on using. We will use the stepper motor to control a sliding door and a servo motor to control the rotating chute. We found some docs for them online to find out how they work and their pin mapping. We also found the Raspberry pi 3 GPIO pinout as thats the controller we are going to use. We got wires and put the pins in the right spot and wrote the code for the two motors. We ran into an issue where there was some confusion on whether we write the GPIO pin number or the pin number but it was quickly resolved by testing them both and using the one that worked. Afterwards both motors worked. The stepper motors spinned clockwise and anticlockwise and the servo motor turned to the angles 45, 90, and 135.

## Feb 07, 2026
**Time: 4:35**

**What we did today:**

Today we worked on the CAD models as a way to represent the project. We started by modelling the box itself in Fusion 360. We wanted it to look like something a large rectangualr prism with three slots on the bottom for each bin that it could go in. In the end we were happy with how it looked and we planned to build it with a wireframe of jinx wood and a body of cardboard.

## Feb 08, 2026
**Time: 1:30**

**What we did today:**

Today we worked on the CAD  model of the sliding door mechanism. It was a 25cm x 25cm sheet of cardboard with a 20cm x 20cm hole in it with 1cm x 1cm holes on the edges so it can be supported by jinx wood. We also decided to use a gear & rack to transfrom the spinning of the stepper motor to linear movement of the door. We choose a gear & reack system as we thought it was the most straightforward system.

## Feb 14, 2026
**Time: 1:45**

**What we did today:**

Today we drew the dimensions onto a carboard template and cut it out to get a top piece with the dimensions of 25cm x 25cm with a circle that has the diameter of 20cm and a the sliding door system below it. Also a big thanks to MR.Springett who printed the gear & rack for us after my printer got clogged. We also hot glued everything together into the right spot. We noticed that some of the cardboard pieces were slightly bigger but in the end they all fit together perfectly after a slight trim. Surprisngly it actually helped a lot that we had a cad model as we could double check dimensions and change them before we actually glue everything in and the fact that if i am not sure where something goes I can just go back and check.