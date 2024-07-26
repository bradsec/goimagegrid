# GOIMAGEGRID

goimagegrid is a Go program that creates image grids or walls from a collection of input images.

## Features

- Parallel image processing for improved performance
- Customisable grid dimensions
- Support for JPEG and PNG image formats
- Intelligent image cropping to find the most interesting areas
- Option to add image names to the grid

## Requirements

- Go 1.20 or higher

## Installing

To install GOIMAGEGRID, you need to have Go installed on your system (https://go.dev/doc/install). Once you have Go installed, you can either clone the repo and run from source or download and install with the following command:

```terminal
go install github.com/bradsec/goimagegrid@latest
```

## Usage

Run the program with the following command:

```
./goimagegrid [flags]
```

### Flags

- `-w` : Width of the grid thumbnail wall (default: 900)
- `-h` : Maximum height for a single grid image (default: 900)
- `-c` : Number of columns in the grid (default: 3)
- `-i` : Directory containing input images (default: "./input")
- `-o` : Directory to save output images (default: "output")
- `-n` : Add image names to the grid images (default: false)

### Samples

**Source images from [Pexels](https://www.pexels.com/)**

![Sample Images](output/nonames/grid_001.jpg)
![Sample Images](output/nonames/grid_002.jpg)
  
With `-n` option adds image names:

![Sample Images](output/grid_001.jpg)
![Sample Images](output/grid_002.jpg)

## License

This project is licensed under the MIT License.
