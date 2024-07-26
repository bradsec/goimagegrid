package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

type ProgressBar struct {
	total      int
	current    int
	width      int
	lastUpdate time.Time
	mu         sync.Mutex
}

func NewProgressBar(total, width int) *ProgressBar {
	return &ProgressBar{
		total:      total,
		width:      width,
		lastUpdate: time.Now(),
	}
}

func (pb *ProgressBar) Update() {
	pb.mu.Lock()
	defer pb.mu.Unlock()
	pb.current++
	if time.Since(pb.lastUpdate) > 100*time.Millisecond {
		pb.Render()
		pb.lastUpdate = time.Now()
	}
}

func (pb *ProgressBar) Render() {
	ratio := float64(pb.current) / float64(pb.total)
	percent := int(ratio * 100)
	filled := int(ratio * float64(pb.width))
	bar := strings.Repeat("█", filled) + strings.Repeat("░", pb.width-filled)
	fmt.Printf("\r[%s] %d%% (%d/%d)", bar, percent, pb.current, pb.total)
}

func printBanner() {
	banner := `
 ██████   ██████                       
██       ██    ██                      
██   ███ ██    ██                      
██    ██ ██    ██                      
 ██████   ██████                       
                                                       
██ ███    ███  █████   ██████  ███████ 
██ ████  ████ ██   ██ ██       ██      
██ ██ ████ ██ ███████ ██   ███ █████   
██ ██  ██  ██ ██   ██ ██    ██ ██      
██ ██      ██ ██   ██  ██████  ███████ 
                                                     
 ██████  ██████  ██ ██████             
██       ██   ██ ██ ██   ██            
██   ███ ██████  ██ ██   ██            
██    ██ ██   ██ ██ ██   ██            
 ██████  ██   ██ ██ ██████ 
`
	fmt.Println(banner)
}

func main() {
	printBanner()

	// Define command-line flags
	var (
		maxGridWidth  int
		maxGridHeight int
		numColumns    int
		imageDir      string
		outputDir     string
		addNames      bool
	)
	flag.IntVar(&maxGridWidth, "w", 900, "Width of the grid thumbnail wall")
	flag.IntVar(&maxGridHeight, "h", 900, "Maximum height for a single grid image")
	flag.IntVar(&numColumns, "c", 3, "Number of columns in the grid")
	flag.StringVar(&imageDir, "i", "./input", "Directory containing input images")
	flag.StringVar(&outputDir, "o", "output", "Directory to save output images")
	flag.BoolVar(&addNames, "n", false, "Add image names to the grid images")
	flag.Parse()

	// Read images
	fmt.Println("Reading images...")
	images, fileNames, err := readImages(imageDir)
	if err != nil {
		log.Fatalf("Error reading images: %v", err)
	}

	numFiles := len(images)
	numRows := numFiles / numColumns
	if numFiles%numColumns != 0 {
		numRows++
	}

	cellWidth := maxGridWidth / numColumns
	cellHeight := cellWidth // Assuming square cells

	// Calculate the number of grids needed
	gridHeight := cellHeight * numRows
	numGrids := (gridHeight + maxGridHeight - 1) / maxGridHeight

	// Create the output directory if it does not exist
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Error creating output directory: %v", err)
	}

	// Create a new progress bar
	progressBar := NewProgressBar(numFiles*2, 50) // Double the total for two-phase progress

	// Phase 1: Reading and resizing images
	resizedImages := processImagesParallel(images, func(img image.Image) image.Image {
		resized := resizeImage(img, uint(cellWidth))
		progressBar.Update()
		return resized
	})

	// Phase 2: Creating grid images
	for gridIndex := 0; gridIndex < numGrids; gridIndex++ {
		startRow := gridIndex * (maxGridHeight / cellHeight)
		endRow := int(math.Min(float64((gridIndex+1)*(maxGridHeight/cellHeight)), float64(numRows)))
		currentGridHeight := (endRow - startRow) * cellHeight

		// Create a new RGBA image for the current grid
		gridImage := image.NewRGBA(image.Rect(0, 0, maxGridWidth, currentGridHeight))

		// Draw images onto the current grid
		for row := startRow; row < endRow; row++ {
			for col := 0; col < numColumns; col++ {
				idx := row*numColumns + col
				if idx >= numFiles {
					break
				}

				resized := resizedImages[idx]
				fileName := fileNames[idx]

				// Calculate the position to draw the image
				x := col * cellWidth
				y := (row - startRow) * cellHeight

				// Draw the resized image onto the gridImage
				draw.Draw(gridImage, image.Rect(x, y, x+cellWidth, y+cellHeight), resized, image.Point{0, 0}, draw.Over)

				// Conditionally add centered image name on top of the image
				if addNames {
					addImageName(gridImage, cellWidth, cellHeight, x, y, fileName)
				}

				// Update progress
				progressBar.Update()
			}
		}

		// Save the current grid image to the output directory
		outputFileName := filepath.Join(outputDir, fmt.Sprintf("grid_%03d.jpg", gridIndex+1))
		if err := saveImage(gridImage, outputFileName); err != nil {
			log.Printf("Error saving image %s: %v", outputFileName, err)
		}
	}

	// Final progress bar update
	progressBar.current = numFiles * 2
	progressBar.Render()

	fmt.Printf("\n\nAll images successfully created in the '%s' directory.\n", outputDir)
}

func processImagesParallel(images []image.Image, processFunc func(image.Image) image.Image) []image.Image {
	var wg sync.WaitGroup
	results := make([]image.Image, len(images))

	for i, img := range images {
		wg.Add(1)
		go func(i int, img image.Image) {
			defer wg.Add(-1)
			results[i] = processFunc(img)
		}(i, img)
	}

	wg.Wait()
	return results
}

func readImages(directory string) ([]image.Image, []string, error) {
	var images []image.Image
	var fileNames []string

	files, err := os.ReadDir(directory)
	if err != nil {
		return nil, nil, err
	}

	for _, file := range files {
		if isImageFile(file) {
			filePath := filepath.Join(directory, file.Name())
			img, err := openImage(filePath)
			if err != nil {
				log.Printf("Error opening image file %s: %v\n", filePath, err)
				continue
			}

			images = append(images, img)
			fileNames = append(fileNames, file.Name())
		}
	}

	// Sort fileNames and reorder images accordingly
	sort.Strings(fileNames)
	sortedImages := make([]image.Image, len(fileNames))

	for i, fileName := range fileNames {
		filePath := filepath.Join(directory, fileName)
		img, err := openImage(filePath)
		if err != nil {
			return nil, nil, fmt.Errorf("error opening image file %s: %v", filePath, err)
		}
		sortedImages[i] = img
	}

	return sortedImages, fileNames, nil
}

func isImageFile(file os.DirEntry) bool {
	ext := strings.ToLower(filepath.Ext(file.Name()))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
}

func openImage(filePath string) (image.Image, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Detect the image format
	_, format, err := image.DecodeConfig(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image config: %v", err)
	}

	// Reset file pointer
	_, err = file.Seek(0, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to reset file pointer: %v", err)
	}

	var img image.Image

	switch format {
	case "jpeg":
		img, err = jpeg.Decode(file)
	case "png":
		img, err = png.Decode(file)
	default:
		return nil, fmt.Errorf("unsupported image format: %s", format)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}

	return img, nil
}

func resizeImage(img image.Image, size uint) image.Image {
	// Get the original image's dimensions
	origBounds := img.Bounds()
	origWidth := origBounds.Dx()
	origHeight := origBounds.Dy()

	// Calculate the scaling factor
	scale := float64(size) / math.Min(float64(origWidth), float64(origHeight))

	// Calculate new dimensions
	newWidth := int(math.Ceil(float64(origWidth) * scale))
	newHeight := int(math.Ceil(float64(origHeight) * scale))

	// Create a new RGBA image
	rect := image.Rect(0, 0, newWidth, newHeight)
	resized := image.NewRGBA(rect)

	// Perform the resize operation
	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			// Map the coordinates to the original image
			origX := int(float64(x) / scale)
			origY := int(float64(y) / scale)

			// Get the color at the mapped coordinates
			c := img.At(origX, origY)

			// Set the color in the new image
			resized.Set(x, y, c)
		}
	}

	// Crop to the most interesting square area
	cropSize := int(size)
	startX, startY := findMostInterestingArea(resized, cropSize)
	cropped := image.NewRGBA(image.Rect(0, 0, cropSize, cropSize))

	for y := 0; y < cropSize; y++ {
		for x := 0; x < cropSize; x++ {
			c := resized.At(startX+x, startY+y)
			cropped.Set(x, y, c)
		}
	}

	return cropped
}

func findMostInterestingArea(img *image.RGBA, size int) (startX, startY int) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	edgeMap := detectEdges(img)
	colorClusters := clusterColors(img)

	maxScore := 0.0
	bestStartX, bestStartY := 0, 0

	centerX := width / 2
	centerY := height / 2
	maxDistance := math.Sqrt(float64(centerX*centerX + centerY*centerY))

	// Slide a window of size x size over the image
	for y := 0; y <= height-size; y++ {
		for x := 0; x <= width-size; x++ {
			colorVariation := calculateColorVariation(img, x, y, size)
			edgeScore := calculateEdgeScore(edgeMap, x, y, size)
			clusterScore := calculateClusterScore(img, colorClusters, x, y, size)

			// Calculate distance from center
			dx := float64(x + size/2 - centerX)
			dy := float64(y + size/2 - centerY)
			distance := math.Sqrt(dx*dx + dy*dy)

			// Create a center weight (1.0 at center, decreasing towards edges)
			centerWeight := 1.0 - (distance / maxDistance)

			// Combine scores (adjust weights as needed)
			totalScore := (colorVariation*0.3 + edgeScore*0.3 + clusterScore*0.2) * (1 + centerWeight)

			if totalScore > maxScore {
				maxScore = totalScore
				bestStartX, bestStartY = x, y
			}
		}
	}

	return bestStartX, bestStartY
}

func detectEdges(img *image.RGBA) [][]float64 {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	edgeMap := make([][]float64, height)
	for i := range edgeMap {
		edgeMap[i] = make([]float64, width)
	}

	for y := 1; y < height-1; y++ {
		for x := 1; x < width-1; x++ {
			gx := colorDifference(img.At(x-1, y), img.At(x+1, y))
			gy := colorDifference(img.At(x, y-1), img.At(x, y+1))
			edgeMap[y][x] = math.Sqrt(gx*gx + gy*gy)
		}
	}

	return edgeMap
}

func colorDifference(c1, c2 color.Color) float64 {
	r1, g1, b1, _ := c1.RGBA()
	r2, g2, b2, _ := c2.RGBA()
	return math.Abs(float64(r1)-float64(r2)) +
		math.Abs(float64(g1)-float64(g2)) +
		math.Abs(float64(b1)-float64(b2))
}

func clusterColors(img *image.RGBA) map[color.Color]int {
	colorCount := make(map[color.Color]int)
	bounds := img.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			c := img.At(x, y)
			colorCount[c]++
		}
	}

	// Keep only the top N most common colors
	N := 10
	type colorFreq struct {
		color color.Color
		count int
	}
	var frequencies []colorFreq
	for c, count := range colorCount {
		frequencies = append(frequencies, colorFreq{c, count})
	}
	sort.Slice(frequencies, func(i, j int) bool {
		return frequencies[i].count > frequencies[j].count
	})

	result := make(map[color.Color]int)
	for i, cf := range frequencies {
		if i >= N {
			break
		}
		result[cf.color] = cf.count
	}
	return result
}

func calculateEdgeScore(edgeMap [][]float64, startX, startY, size int) float64 {
	var score float64
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			score += edgeMap[startY+y][startX+x]
		}
	}
	return score / float64(size*size)
}

func calculateClusterScore(img *image.RGBA, clusters map[color.Color]int, startX, startY, size int) float64 {
	colorCount := make(map[color.Color]int)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			c := img.At(startX+x, startY+y)
			if _, ok := clusters[c]; ok {
				colorCount[c]++
			}
		}
	}

	var score float64
	for _, count := range colorCount {
		score += math.Log(float64(count + 1))
	}
	return score / float64(size*size)
}

func calculateColorVariation(img *image.RGBA, startX, startY, size int) float64 {
	var sumR, sumG, sumB, sumRSq, sumGSq, sumBSq float64
	count := float64(size * size)

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, b, _ := img.At(startX+x, startY+y).RGBA()
			rF, gF, bF := float64(r>>8), float64(g>>8), float64(b>>8)

			sumR += rF
			sumG += gF
			sumB += bF
			sumRSq += rF * rF
			sumGSq += gF * gF
			sumBSq += bF * bF
		}
	}

	// Calculate variance for each channel
	varR := (sumRSq - (sumR * sumR / count)) / count
	varG := (sumGSq - (sumG * sumG / count)) / count
	varB := (sumBSq - (sumB * sumB / count)) / count

	// Return the sum of variances as a measure of color variation
	return varR + varG + varB
}

func addImageName(img *image.RGBA, cellWidth, cellHeight, x, y int, name string) {
	textHeight := 20
	textImg := image.NewRGBA(image.Rect(0, 0, cellWidth, textHeight))

	// Draw background rectangle for text
	draw.Draw(textImg, textImg.Bounds(), &image.Uniform{color.RGBA{0, 0, 0, 180}}, image.Point{}, draw.Src)

	// Draw centered text on the background
	textColor := color.White
	drawCenteredText(textImg, name, cellWidth, textHeight, textColor)

	// Draw the text image onto the main image
	draw.Draw(img, image.Rect(x, y+cellHeight-textHeight, x+cellWidth, y+cellHeight), textImg, image.Point{0, 0}, draw.Over)
}

func drawCenteredText(img *image.RGBA, text string, width, height int, c color.Color) {
	face := basicfont.Face7x13
	maxWidth := width - 10 // Leave a 5-pixel margin on each side

	// Check if the text fits
	textWidth := font.MeasureString(face, text).Ceil()
	if textWidth > maxWidth {
		// If it doesn't fit, truncate and add ellipsis
		text = truncateText(text, face, maxWidth)
	}

	// Recalculate text width after potential truncation
	textWidth = font.MeasureString(face, text).Ceil()
	x := (width - textWidth) / 2
	y := (height + face.Metrics().Ascent.Ceil()) / 2

	point := fixed.Point26_6{X: fixed.Int26_6(x * 64), Y: fixed.Int26_6(y * 64)}
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: face,
		Dot:  point,
	}
	d.DrawString(text)
}

func truncateText(text string, face font.Face, maxWidth int) string {
	ellipsis := "..."
	ellipsisWidth := font.MeasureString(face, ellipsis).Ceil()

	// If even the ellipsis doesn't fit, return an empty string
	if ellipsisWidth > maxWidth {
		return ""
	}

	// Truncate the text
	for len(text) > 0 {
		width := font.MeasureString(face, text).Ceil()
		if width+ellipsisWidth <= maxWidth {
			return text + ellipsis
		}
		text = text[:len(text)-1]
	}

	return ellipsis
}

func saveImage(img image.Image, filename string) error {
	outputFile, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating output file: %v", err)
	}
	defer outputFile.Close()

	return jpeg.Encode(outputFile, img, nil)
}
