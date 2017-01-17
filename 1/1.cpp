#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>
#include <map>
#include <cstdint>

const uint8_t null = 0x00;

struct bstream : public std::fstream {
    bstream(const char *filename, ios_base::openmode mode): std::fstream(filename, mode) {}

    /* Overload the >> operator to perform a reinterpret cast.
     * In other words, performs a cast of a series of bytes equal to
     * the size of the operand.
     */
    template<typename T> bstream& operator>>(const T& value) {
        read(reinterpret_cast<char*>(const_cast<T*>(&value)), sizeof(T));
        return *this;
    }
    bstream& operator>>(unsigned char& c) {
        read(reinterpret_cast<char*>(&c), 1);
        return *this;
    }

    /* Overload the << operator to perform a reinterpret cast.
     * In other words, performs a cast of a series of bytes equal to
     * the size of the operand.
     */
    template<typename T> bstream& operator<<(const T& value) {
        write(reinterpret_cast<const char*>(&value), sizeof(T));
        return *this;
    }
};

struct BitMapFile
{
public:
    /* BITMAP FILE HEADER
     * Fields:
     *   - SIGNATURE (2 byte string) at 0x00
     *   - FILE SIZE (4 byte integer) at 0x02
     *   - PIXELARRAY OFFSET (4 byte integer) at 0x0a
     */
    std::string signature       ;
    uint32_t fileSize           ;
    uint32_t pixelArrayOffset   ;
    /* BitMapINFOHEADER
     * Fields:
     *   - DIB HEADER SIZE (4 byte integer) at 0x0e
     *   - BitMap width (4 byte integer) at 0x12
     *   - BitMap height (4 byte integer) at 0x16
     *   - PLANES (2 byte integer) at 0x
     *   - DEPTH (2 byte integer) at 0x
     *   - COMPRESSION METHOD (4 byte integer) at 0x
     *   - IMAGE SIZE (4 byte integer) at 0x
     *   - HORIZONTAL RESOLUTION (4 byte integer) at 0x
     *   - VERTICAL RESOLUTION (4 byte integer) at 0x
     *   - NUMBER OF COLORS IN PALETTE (4 byte integer) at 0x
     *   - NUMBER OF IMPORTANT COLORS (4 byte integer) at 0x
     */
    uint32_t DIBHeaderSize  ;
    uint32_t width          ;
    uint32_t height         ;
    uint16_t planes         ;
    uint16_t depth          ;
    uint32_t compression    ;
    uint32_t imageSize      ;
    uint32_t horizontalRes  ;
    uint32_t verticalRes    ;
    uint32_t colorPalette   ;
    uint32_t importantColors;

    BitMapFile(): signature("BM") {}
    BitMapFile(bstream& is): signature("BM") {
        is.seekg(0x02); is >> fileSize              ;
        is.seekg(0x0a); is >> pixelArrayOffset      ;
        is >> DIBHeaderSize                                             // Read BitMapINFOHEADER
           >> width
           >> height
           >> planes
           >> depth
           >> compression
           >> imageSize
           >> horizontalRes
           >> verticalRes
           >> colorPalette
           >> importantColors;
    }

    void printHeader() {
        std::cout << " ------------------------ \t ------------------------\n"
                     " |  BitMap FILE HEADER  | \t |  BitMap FILE HEADER  |\n"
                     " ------------------------ \t ------------------------\n"
                     " | SIGNATURE |            \t |\t" << signature << "\t|\n"
                     " |      FILE SIZE       | \t |\t" <<  fileSize << "\t\t|\n"
                     " | RESERVED  | RESERVED | \t | RESERVED  | RESERVED |\n"
                     " | OFFSET TO PIXELARRAY | \t |\t" << pixelArrayOffset << "\t\t|\n"
                     " ------------------------ \t ------------------------\n"
                  << std::endl
                  << " ------------------------ \t ------------------------\n"
                     " |   BitMapINFOHEADER   | \t |   BitMapINFOHEADER   |\n"
                     " ------------------------ \t ------------------------\n"
                     " |    DIB HEADER SIZE   | \t |\t" << DIBHeaderSize << "\t\t|\n"
                     " |     BITMAP WIDTH     | \t |\t" << width << "\t\t|\n"
                     " |     BITMAP HEIGHT    | \t |\t" << height << "\t\t|\n"
                     " |  PLANES  |   DEPTH   | \t |  " << planes << "\t |\t"<< depth << "\t|\n"
                     " |  COMPRESSION METHOD  | \t |\t" << compression <<"\t\t|\n"
                     " |      IMAGE SIZE      | \t |\t" << imageSize << "\t\t|\n"
                     " | HORIZONTAL RESOLUTION| \t |\t" << horizontalRes << "\t\t|\n"
                     " |  VERTICAL RESOLUTION | \t |\t" << verticalRes << "\t\t|\n"
                     " |  COLOURS IN PALETTE  | \t |\t" << colorPalette << "\t\t|\n"
                     " |   IMPORTANT COLORS   | \t |\t" << importantColors << "\t\t|\n"
                     " ------------------------ \t ------------------------\n"
                  << std::endl;
    }
};

struct ColourBitMapFile : public BitMapFile {
    typedef std::tuple<uint8_t, uint8_t, uint8_t> Pixel;
    typedef std::vector<std::vector<Pixel>> BitMap;
    typedef BitMap::size_type bitmap_sz;

    BitMap bitmap;

    ColourBitMapFile(bstream& is)
    : BitMapFile(is), bitmap(height, std::vector<Pixel>(width)) {
        if (depth != 24)
            throw std::domain_error("Input file is a grayscale image");
        for (bitmap_sz i = 0; i != height; ++i) {
            for (bitmap_sz j = 0; j != width; ++j) {
                uint8_t blue, green, red;
                is >> blue >> green >> red;
                bitmap[height - 1 - i][j] = std::tie(red, green, blue);
            }
        }
    }
};

struct GrayScaleBitMapFile : public BitMapFile {
    typedef uint8_t Pixel;
    typedef std::vector<std::vector<Pixel>> BitMap;
    typedef BitMap::size_type bitmap_sz;

    BitMap bitmap;

    GrayScaleBitMapFile(bstream& is)
    : BitMapFile(is), bitmap(height, std::vector<Pixel>(width)) {
        for (bitmap_sz i = 0; i != height; ++i)
            for (bitmap_sz j = 0; j != width; ++j)
                is >> bitmap[height - 1 - i][j];
    }

    GrayScaleBitMapFile(ColourBitMapFile bmp)
    : BitMapFile(), bitmap(bmp.height, std::vector<Pixel>(bmp.width)) {
        uint16_t paletteSize = 4 * 256;                                 // 256 colours * 4 channel
        fileSize        = (bmp.fileSize - bmp.imageSize) + bmp.imageSize/3 + paletteSize;
        pixelArrayOffset= bmp.pixelArrayOffset + paletteSize;

        DIBHeaderSize   = bmp.DIBHeaderSize;
        width           = bmp.width;
        height          = bmp.height;
        planes          = bmp.planes;
        depth           = bmp.depth/3;
        compression     = bmp.compression;
        imageSize       = bmp.imageSize/3;
        horizontalRes   = bmp.horizontalRes;
        verticalRes     = bmp.verticalRes;
        colorPalette    = bmp.colorPalette;
        importantColors = bmp.importantColors;

        for (bitmap_sz i = 0; i != height; ++i){                                    // Convert 24-bit image to grayscale
            for (bitmap_sz j = 0; j != width; ++j){
                GrayScaleBitMapFile::Pixel red, green, blue;
                std::tie(red, green, blue) = bmp.bitmap[i][j];
                bitmap[i][j] = (red * 0.2126 + green * 0.7152 + blue * 0.0722);     // BT.709 specification
            }
        }
    }

    void transpose() {
        for (bitmap_sz i = 0; i != height; ++i)
            for (bitmap_sz j = 0; j != i; ++j)
                std::swap(bitmap[i][j], bitmap[j][i]);
        std::swap(height, width);
        std::swap(horizontalRes, verticalRes);
    }

    bstream& save(bstream& os) {
        os.seekp(0x00); os.write(&signature[0], 2);                                 // Write the BITMAPFILEHEADER
        os.seekp(0x02); os << fileSize           ;
        os.seekp(0x0a); os << pixelArrayOffset   ;
        os << DIBHeaderSize << width << height << planes << depth << compression    // Write the BitMapINFOHEADER
           << imageSize << horizontalRes << verticalRes << colorPalette << importantColors;
        os << null << null << null << null;                                         // Write the grayscale colour palette
        for(uint8_t i = 1; i != 0; ++i)
            os << i << i << i << null;
        for (bitmap_sz i = 0; i != height; ++i)
            for (bitmap_sz j = 0; j != width; ++j)
                os << bitmap[height - 1 - i][j];
        return os;
    }
};

ColourBitMapFile& readBMP(char* input)
{
    bstream is(input, std::ios::in|std::ios::binary);                         // Open the input BMP file stream
    if (!is.is_open())                                                        // Check if file can be opened
        throw std::invalid_argument("Couldn't open input BitMap file");
    ColourBitMapFile* colourBMP = new ColourBitMapFile(is);
    colourBMP->printHeader();
    is.close();
    return *colourBMP;
}

GrayScaleBitMapFile& convertFlipGrayScale(ColourBitMapFile& colourBMP)
{
    GrayScaleBitMapFile* grayscaleBMP = new GrayScaleBitMapFile(colourBMP);
    grayscaleBMP->printHeader();
    grayscaleBMP->transpose();
    return *grayscaleBMP;
}

void writeBMP(char* output, GrayScaleBitMapFile& grayscaleBMP)
{
    bstream os(output, std::ios::out|std::ios::binary);                        // Open the output BMP file stream
    grayscaleBMP.save(os);
    os.close();
}


int main(int argc, char* argv[])
{
    if (argc != 3) {                                                            // Check for commandline arguments
        std::cout << "Incorrect arguments\n"
                     "Usage: ./a.out <input image path> <output image path>\n"
                     "For eg: ./a.out lena.bmp out.bmp";
        return -1;
    }

    try {
        ColourBitMapFile colourBMP = readBMP(argv[1]);
        GrayScaleBitMapFile grayscaleBMP = convertFlipGrayScale(colourBMP);
        writeBMP(argv[2], grayscaleBMP);
    }
    catch(...) {
        std::cout << "Error reading BMP file" <<std::endl;
        return -1;
    }
    std::cout << "Enter a key to exit" << std::endl;
    std::string temp;
    std::cin >> temp;
    return 0;
}