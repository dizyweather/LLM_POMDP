# LLM_POMDP

Hello to whoever is seeing this!

We have two main folders, one with the code and evaulations of the TigerPOMDP problem and another for the ExplorePOMDP problem.
For the TigerPOMDP, run tiger_problem.py. You can change the settings of the POMDP problem at the top in POMDP_CONFIG, and change which model you are querying at the bottom of the program in main.

For the ExplorePOMDP, we have Explore.py where in the main function you can do the same in changing amount of queries and which model.
There's also the arenas.py which is where I generate the random arenas for the LLMs to work on. You can change the spawn rate of items in that file.

If you want to run, you'll have to make a .env file with:

```
GEMINI_API_KEY=cool_api_key
```

to send requests to the model.
