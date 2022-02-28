# FragmentationNavigation
Inspired by a paper on spatial navigation and map fragmentation from the Fiete Lab, MIT.

User may choose either online or offline navigation of an environment they create using pygame.

In offline navigation, the environment is charted and evaluated for distinct rooms and locations within it. These locations are displayed in the form of a clustered isomap.

In online navigation, the environment is produced by the user, and the user may construct a path of their own choosing through the environment. The program takes note of distinct locations separated by bottleneck regions and stores these in memory.

Linked here is the paper this program draws its concepts from: 
https://www.biorxiv.org/content/10.1101/2021.10.29.466499v2

Other sources I used to inform the development of this code: 
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2677716/#FD1
https://edoc.ub.uni-muenchen.de/26577/1/Nagele_Johannes.pdf
https://numenta.com/blog/2018/05/25/how-grid-cells-map-space/
