# Movie Director's Office
---
- **topic**: Investment opportunity assessment
- **client**: [Sequoia](https://www.sequoiacap.com/)
- **oppotunity**: [Universal Pictures](https://www.universalpictures.com/) has offered Sequoia to buy options for the movie rights of the upcoming movies of their Dark Universe franchise.
---

## Scope of the project

The Dark Universe is a franchise of monster movies produced by Universal Pictures. If the concept was promising, latest release such as `Dracula Untold` were not as successful as expected. The franchise seems in difficulty as the production studio has scrapped Bride of Frankenstein, Frankenstein's Monster, The Invisible Man, Dr. Jekyll & Mr. Hyde, Dracula Untold 2, and a remake of Van Helsing from its production schedule. The studio is willing to share the risk and potential reward with Sequoia buy selling bundled options for the upcoming movies rights.

Sequoia has a strong expertise in risky ventures and movies are risky ventures for sure. However, the studio has no track record in the movie industry and is looking for an external assessment of the opportunity.

Sequoia has reached out to the `Movie Director's Office` consultancy firm to answer the following question:

> Is it possible to predict the success of a movie before its release?

___

This case study has been realised in the context of the Hackathon course at the [Vlerick Business School](https://www.vlerick.com/en/programmes/masters-programmes/masters-in-general-management-business-analytics-ai-track/).

The Hackathon is centred on the principle of innovation and here are a few technical innovation which have been used to realise the case study:

- `Object-oriented programming` and `Encapsulation`: all the code for preparation, vizualisation, and modelling has been encapsulated in classes and packaged in the MDO package to make the final report accessible to non-technical profiles. All the source code is available in the movie_director_office.py module.

- `Interoperability`: to enable reuse (GPU License), the code has been documented, saved with version controlling on the [hackathon repository](https://github.com/MathieuDemarets/hackathon). The packages used to develop this project have also been saved in the mdo_requirements text file.

- `Multi-threaded webscrapping`: to enrich the data with the release years of the various movies, we created a bot to browse, search the movie, and retrieve the year of release. That bot can work in parallel with other bots to speed up the process.

- `API calls`: to simulate the agent's knowledge of the celebrities, we gave it access to an API giving information on the actors.

- `Coding best practices`: all classes have been documented with numpy docstring, types have been defined to support mypy type checking, and the style has been adapted through black, and checked through pylint to reach a minimum of 7.5/10 (to allow some flexibility).

- `Geospatial analysis`: geospatial data has been used to enrich the dataset (simple index matching) but also to create an interactive world map for easy vizualisation and enabling the user to get the information he/she wants.

- `Network analysis`: we reformatted the actors name and movie name features to build a graph of actors who played together. Then we applied graph theory to engineer new features on the base of this relationship network.

I hope you enjoyed reading this case study as much as I had fun thinking about the narrative and the various techniques to leverage.

[Mathieu Demarets](https://www.linkedin.com/in/mathieudemarets/)