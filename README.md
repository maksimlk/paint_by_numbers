# Painting By Numbers Conversion

This is a python script that converts any image to an
outline with each segment numbered according to a color. It produces the outline, color palette with all the colors used, and a resulting image (see examples).

## Dependencies

You will need all the modules indicated in requirements.txt

```bash
pip install -r requirements.txt
```

List of dependencies:
```
opencv-python~=4.9.0.80
numpy~=2.0.0rc2
matplotlib~=3.9.0
pillow~=10.3.0
shapely~=2.0.4
```
## Usage

```bash
python paint_by_numbers.py -i ./input_image.jpg
```

Additional arguments:
-o name and directory for the output files. Default is "./output/output.jpg"
-w width of the output image (height is scaled automatically). Default 1024
-k number of colors to be used (number of clusters for kmeans algorithm). Default is 16
-b blur of gaussian blur to be applied before KMeans (smaller value results in less segments). Default is 20
-ma to set the minimum value for polygon area (polygons with smaller area will be merged with the largest neighbour). Default is 1 (relative, try different values)

## Examples

Converted images:

Baboon

![Example1](https://imgur.com/AyV4EeH.png)
![Example1](https://imgur.com/w1zxp3O.png)
![Example1](https://imgur.com/9MymMxe.png)

The Tower of Babel (Bruegel)

![Example2](https://i.imgur.com/ZIVQXke.png)
![Example2](https://imgur.com/br0tCyC.png)
![Example2](https://imgur.com/6qEKM7d.png)

Stańczyk (Matejko)

![Example3](https://imgur.com/H0pKjsS.png)
![Example3](https://imgur.com/6WXTXVg.png)
![Example3](https://imgur.com/BV3nzaq.png)

## Roadmap

- [x] Create working script to preprocess image, quantize colors using KMeans, and convert segments of different colours to shapely polygons.
- [x] Merge small polygons together.
- [x] Number polygons in the outline.
- [ ] Improve complexity by dividing image into pieces and merging small polygons in those pieces first.
- [ ] Split up large narrow polygons.
- [ ] Create GUI, possibly on top of flask as a website
