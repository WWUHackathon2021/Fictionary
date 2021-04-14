# Fictionary

Our team created a web application that generates coherent definitions fictitious for words. We created this project in less than 24 hours during the 2021 Western Washington University Hackathon and won 1st place in the category for students starting without code.

The app consists of a language model, a website, and an API that passes information between the two.

We fine-tuned the language model GPT-2 on the words and corresponding definitions in the dictionary in order to train it to generate new definitions given a word. We compared the results between fine-tuning on Urban Dictionary and [a CSV version of the Online Plain Text English Dictionary](https://www.bragitoff.com/2016/03/english-dictionary-in-csv-format/) and, though both datasets worked, we decided to go with Urban Dictionary as that dataset was significantly larger and provided us with more fun definitions. In order to make the definitions generated from Urban Dictionary less vulgar, we applied a [profanity check](https://github.com/vzhou842/profanity-check) to the data, which paired the dataset down to around 2 million word-definition pairs. The model currently in-use for the API is trained on a smaller subset of that data. After conducting more testing, we've found that this has not removed all explicit material, and as a result **please be aware that the model can be quite vulgar at times**.

We coded the website with Node.js and Express for the webserver backend, handlebars for HTML templating, and Sass for styling. We used the Twitter API to implement a button for tweeting about Fictionary to your friends. We used the Zazzle API to implement a button for buying a mug that displays your word and definition.

We coded an API using the Flask Python framework that we use to communicate between the model and website.

logan testing, delete right after
