#include <iostream>
#include <fstream>
#include <vector>
#include <map>

using std::ifstream;
using std::ofstream;

const uint8_t null = 0x00;
const int DIB_HEADER_OFFSET = 0x0e;
const std::map<uint32_t, std::string> COMPRESSION_METHODS = {
    {0, "BI_RGB"},
    {1, "BI_RLE8"},
    {2, "BI_RLE4"},
    {3, "BI_BITFIELDS"},
    {4, "BI_JPEG"},
    {5, "BI_PNG"},
    {6, "BI_ALPHABITFIELDS"},
    {11,"BI_CMYK"},
    {12,"BI_CMYKRLE8"},
    {13,"BI_CMYKRLE4"},
};

class ibstream : public ifstream {
public:
    using ifstream::ifstream;

    /* Overload the >> operator to perform a reinterpret cast.
     * In other words, performs a cast of a series of bytes equal to
     * the size of the operand.
     */
    template<typename T>
    ibstream& operator>>(const T& value) {
        this->read(reinterpret_cast<char*>(const_cast<T*>(&value)), sizeof(T));
        return *this;
    }

    ibstream& operator>>(unsigned char& c) {
        this->read(reinterpret_cast<char*>(&c), 1);
        return *this;
    }
};

class obstream : public ofstream {
public:
    using ofstream::ofstream;

    /* Overload the << operator to perform a reinterpret cast.
     * In other words, performs a cast of a series of bytes equal to
     * the size of the operand.
     */
    template<typename T>
    obstream& operator<<(const T& value) {
        this->write(reinterpret_cast<const char*>(&value), sizeof(T));
        return *this;
    }
};


int main(int argc, char* argv[])
{
    if (argc != 3) {                                                            // Check for commandline arguments
        std::cerr << "Incorrect arguments\n"
                     "Usage: ./a.out <input image path> <output image path>\n";
        return 1;
    }

    ibstream is(argv[1], std::ios::in|std::ios::binary);                        // Open the input BMP file stream
    if (!is.is_open()) {                                                        // Check if file can be opened
        std::cerr << "Couldn't open input file";
        return 1;
    }

    /* BITMAP FILE HEADER
     * Fields:
     *   - SIGNATURE (2 byte string) at 0x00
     *   - FILE SIZE (4 byte integer) at 0x02
     *   - PIXELARRAY OFFSET (4 byte integer) at 0x0a
     */
    std::string SIGNATURE(2, 0) ; is.seekg(0x00); is.read(&SIGNATURE[0], 2) ;
    uint32_t FILE_SIZE          ; is.seekg(0x02); is >> FILE_SIZE           ;
    uint32_t PIXELARRAY_OFFSET  ; is.seekg(0x0a); is >> PIXELARRAY_OFFSET   ;

    std::cout << " ------------------------ \t ------------------------\n"
                 " |  BITMAP FILE HEADER  | \t |  BITMAP FILE HEADER  |\n"
                 " ------------------------ \t ------------------------\n"
                 " | SIGNATURE |            \t |\t" << SIGNATURE << "\t|\n"
                 " |      FILE SIZE       | \t |\t" <<  FILE_SIZE << "\t\t|\n"
                 " | RESERVED  | RESERVED | \t | RESERVED  | RESERVED |\n"
                 " | OFFSET TO PIXELARRAY | \t |\t" << PIXELARRAY_OFFSET << "\t\t|\n"
                 " ------------------------ \t ------------------------\n"
              << std::endl;

    /* BITMAPINFOHEADER
     * Fields:
     *   - DIB HEADER SIZE (4 byte integer) at 0x0e
     *   - BITMAP WIDTH (4 byte integer) at 0x12
     *   - BITMAP HEIGHT (4 byte integer) at 0x16
     *   - PLANES (2 byte integer) at 0x
     *   - DEPTH (2 byte integer) at 0x
     *   - COMPRESSION METHOD (4 byte integer) at 0x
     *   - IMAGE SIZE (4 byte integer) at 0x
     *   - HORIZONTAL RESOLUTION (4 byte integer) at 0x
     *   - VERTICAL RESOLUTION (4 byte integer) at 0x
     *   - NUMBER OF COLORS IN PALETTE (4 byte integer) at 0x
     *   - NUMBER OF IMPORTANT COLORS (4 byte integer) at 0x
     */
    is.seekg(DIB_HEADER_OFFSET);                                                // Move input stream to DIB HEADER
    uint32_t DIB_HEADER_SIZE; is >> DIB_HEADER_SIZE ;
    uint32_t WIDTH          ; is >> WIDTH           ;
    uint32_t HEIGHT         ; is >> HEIGHT          ;
    uint16_t PLANES         ; is >> PLANES          ;
    uint16_t DEPTH          ; is >> DEPTH           ;
    uint32_t COMPRESSION    ; is >> COMPRESSION     ;
    uint32_t IMG_SIZE       ; is >> IMG_SIZE        ;
    uint32_t HORIZONTAL_RES ; is >> HORIZONTAL_RES  ;
    uint32_t VERTICAL_RES   ; is >> VERTICAL_RES    ;
    uint32_t COLOR_PALETTE  ; is >> COLOR_PALETTE   ;
    uint32_t IMP_COLORS     ; is >> IMP_COLORS      ;

    std::cout << " ------------------------ \t ------------------------\n"
                 " |   BITMAPINFOHEADER   | \t |   BITMAPINFOHEADER   |\n"
                 " ------------------------ \t ------------------------\n"
                 " |    DIB HEADER SIZE   | \t |\t" << DIB_HEADER_SIZE << "\t\t|\n"
                 " |     BITMAP WIDTH     | \t |\t" << WIDTH << "\t\t|\n"
                 " |     BITMAP HEIGHT    | \t |\t" << HEIGHT << "\t\t|\n"
                 " |  PLANES  |   DEPTH   | \t |  " << PLANES << "\t |\t"<< DEPTH << "\t|\n"
                 " |  COMPRESSION METHOD  | \t |\t" << COMPRESSION_METHODS.at(COMPRESSION) <<"\t\t|\n"
                 " |      IMAGE SIZE      | \t |\t" << IMG_SIZE << "\t\t|\n"
                 " | HORIZONTAL RESOLUTION| \t |\t" << HORIZONTAL_RES << "\t\t|\n"
                 " |  VERTICAL RESOLUTION | \t |\t" << VERTICAL_RES << "\t\t|\n"
                 " |  COLOURS IN PALETTE  | \t |\t" << COLOR_PALETTE << "\t\t|\n"
                 " |   IMPORTANT COLORS   | \t |\t" << IMP_COLORS << "\t\t|\n"
                 " ------------------------ \t ------------------------\n"
              << std::endl;

    is.seekg(PIXELARRAY_OFFSET);                                                // Move input stream to PIXEL ARRAY

    // // Dynamically allocate a single channel matrix
    std::vector<std::vector<uint8_t>> grayscale (HEIGHT, std::vector<uint8_t>(WIDTH));
    typedef std::vector<std::vector<uint8_t>>::size_type vec_sz;

    if (DEPTH == 8) {
        for (vec_sz i = 0; i != HEIGHT; ++i)
            for (vec_sz j = 0; j != WIDTH; ++j)
                is >> grayscale[HEIGHT - 1 - i][j];
    } else {
        // Dynamically allocate 3-channel matrices
        std::vector<std::vector<uint8_t>> red   (HEIGHT, std::vector<uint8_t>(WIDTH));
        std::vector<std::vector<uint8_t>> green (HEIGHT, std::vector<uint8_t>(WIDTH));
        std::vector<std::vector<uint8_t>> blue  (HEIGHT, std::vector<uint8_t>(WIDTH));
        char padding[8];                                                        // Temporary string to store padding

        std::cout << "Image Pixel Array (R,G,B):" << std::endl;
        for (vec_sz i = 0; i != HEIGHT; ++i)                                    // Raster scan
            for (vec_sz j = 0; j != WIDTH; ++j)
                is >> blue[HEIGHT - 1 - i][j] >> green[HEIGHT - 1 - i][j] >> red[HEIGHT - 1 - i][j];

        for (vec_sz i = 0; i != HEIGHT; ++i) {
            for (vec_sz j = 0; j != WIDTH; ++j) {
                if (i <= 3 && j <= 3)
                    // Padding?
                    std::cout << "(" << +red[i][j] << "," << +green[i][j] << "," << +blue[i][j] << ")\t";
            }
            if (i <= 3)
                std::cout << "..." << std::endl;
        }
        std::cout << "." << std::endl << "." << std::endl << ".\t\t\t\t\t"
                  << HEIGHT << " rows * " << WIDTH << " columns" << std::endl;

        for (vec_sz i = 0; i != HEIGHT; ++i)                                    // Convert 24-bit image to grayscale
            for (vec_sz j = 0; j != WIDTH; ++j)
                grayscale[i][j] = (red[i][j]*0.2126 + green[i][j]*0.7152 + blue[i][j]*0.0722);   // BT.709 specification

        // Update header specification
        int PALETTE_SIZE = 4 * 256;
        FILE_SIZE = (FILE_SIZE - IMG_SIZE) + IMG_SIZE/3 + PALETTE_SIZE;
        DEPTH = DEPTH/3;
        IMG_SIZE = IMG_SIZE/3;
        PIXELARRAY_OFFSET += PALETTE_SIZE;
    }

    std::cout << "Grayscale Pixel Array (I):" << std::endl;                     // Print Grayscale Image
    for (vec_sz i = 0; i != HEIGHT; ++i) {
        for (vec_sz j = 0; j != WIDTH; ++j) {
            if (i <= 3 && j <= 3)
                std::cout << "(" << +grayscale[i][j] << ")\t";
        }
        if (i <= 3)
            std::cout << "..." << std::endl;
    }
    std::cout << "." << std::endl << "." << std::endl << ".\t\t\t\t\t"
              << HEIGHT << " rows * " << WIDTH << " columns" << std::endl;

    obstream os(argv[2], std::ios::out|std::ios::binary);                       // Create output BMP file

    os.seekp(0x00); os.write(&SIGNATURE[0], 2);                                 // Write the Bitmap File Header
    os.seekp(0x02); os << FILE_SIZE           ;
    os.seekp(0x0a); os << PIXELARRAY_OFFSET   ;

    os << DIB_HEADER_SIZE << HEIGHT << WIDTH << PLANES <<DEPTH << COMPRESSION
       << IMG_SIZE << VERTICAL_RES << HORIZONTAL_RES << COLOR_PALETTE << IMP_COLORS;

    os << null << null << null << null;
    for(uint8_t i = 1; i != 0; ++i) {
        os << i << i << i << null;
    }

    for(vec_sz j = 0; j != WIDTH; ++j) {                                        // Iterate in a transpose fashion
        for (vec_sz i = 0; i != HEIGHT; ++i) {
            os << grayscale[i][WIDTH - 1 - j];
        }
    }
    is.close();
    os.close();
    return 0;
}