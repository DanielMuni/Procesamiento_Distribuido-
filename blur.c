#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <malloc.h>
#include "omp.h"
#include <string.h>

// Standard values located at the header of an BMP file
#define MAGIC_VALUE    0X4D42 
//Bit depth
#define BITS_PER_PIXEL 24
#define NUM_PLANE      1
#define COMPRESSION    0
#define BITS_PER_BYTE  8
//OpenMP var
#define NUM_THREADS 200

#pragma pack(1)

/*Section used to declare structures*/
typedef struct{
  uint16_t type;
  uint32_t size;
  uint16_t reserved1;
  uint16_t reserved2;
  uint32_t offset;
  uint32_t header_size;
  uint32_t width;
  uint32_t height;
  uint16_t planes;
  uint16_t bits;
  uint32_t compression;
  uint32_t imagesize;
  uint32_t xresolution;
  uint32_t yresolution;
  uint32_t importantcolours;
}BMP_Header;

typedef struct{
  BMP_Header header;
  unsigned int pixel_size;
  unsigned int width;
  unsigned int height;
  unsigned int bytes_per_pixel;
  unsigned char * pixel; //For future allocation in memory
}BMP_Image;

/*Section used to declare functions*/
int checkHeader(BMP_Header *);
BMP_Image* cleanUp(FILE *, BMP_Image *);
BMP_Image* BMP_open(const char *);
int BMP_save(const BMP_Image *img, const char *filename);
void BMP_destroy(BMP_Image *img);


/*End section*/

int checkHeader(BMP_Header *hdr){
  if((hdr -> type) != MAGIC_VALUE)           {printf("No es un bmp\n"); return 0;}
  if((hdr -> bits) != BITS_PER_PIXEL)        {printf("Revisa bit depth\n"); return 0;}
  if((hdr -> planes) != NUM_PLANE)           {printf("Array de diferente dimensiones\n"); return 0;}
  if((hdr -> compression) != COMPRESSION)    {printf("Hay compresion\n"); return 0;}
  return 1;
}

/*Funtion used as cleaner, in incorrect format of an image*/
BMP_Image * cleanUp(FILE * fptr, BMP_Image * img)
{
  if (fptr != NULL)
    {
      fclose (fptr);
    }
  if (img != NULL)
    {
      if (img -> pixel != NULL)
	{
	  free (img -> pixel);
	}
      free (img);
    }
  return NULL;
}

BMP_Image* BMP_open(const char *filename){
  FILE *fptr = NULL;
  BMP_Image *img = NULL;
  fptr = fopen(filename, "rb");
  if(fptr == NULL){printf("Archivo no existe\n"); return cleanUp(fptr,img);}
  img = malloc(sizeof(BMP_Image));
  if(img == NULL){return cleanUp(fptr,img);}
  if(fread(&(img -> header), sizeof(BMP_Header),1,fptr) != 1) {printf("Header no disponible\n"); return cleanUp(fptr,img);}
  if(checkHeader(&(img -> header)) == 0) {printf("Header fuera del estandar\n"); return cleanUp(fptr,img);}
  img -> pixel_size      = (img -> header).size - sizeof(BMP_Header);
  img -> width           = (img -> header).width;
  img -> height          = (img -> header).height;
  img -> bytes_per_pixel = (img -> header).bits/BITS_PER_BYTE;
  img -> pixel = malloc(sizeof(unsigned char) * (img -> pixel_size));
  if((img -> pixel) == NULL){printf("Imagen vacia\n"); return cleanUp(fptr,img);}
  if(fread(img->pixel, sizeof(char), img -> pixel_size,fptr) != (img -> pixel_size)){printf("Imagen con contenido irregular \n");return cleanUp(fptr,img);}
  char onebyte;
  if(fread(&onebyte,sizeof(char),1,fptr) != 0) {printf("Hay pixeles residuales\n"); return cleanUp(fptr,img);}
  fclose(fptr);
  return img;
}

int BMP_save(const BMP_Image *img, const char *filename){
  FILE *fptr = NULL;
  fptr = fopen(filename, "wb");
  if(fptr == NULL) {return 0;}//Maybe you should write the header first
  if(fwrite(&(img -> header), sizeof(BMP_Header),1,fptr) != 1) {fclose(fptr); return 0;}
  if(fwrite(img->pixel, sizeof(char), img -> pixel_size, fptr) != (img -> pixel_size)) {fclose(fptr); return 0;}
  fclose(fptr);
  return 1;
}

void BMP_destroy(BMP_Image *img){
  free (img -> pixel);
  free (img);
}

void specs(BMP_Image* img){
  printf("Image width: %i\n", img->width);
  printf("Image height: %i\n", abs(img->height));
  printf("Image BPP: %i\n",  img->bytes_per_pixel);
  printf("Image size: %i\n",  img->pixel_size);
}



/*Debugging functions*/
float ** kernel(unsigned int size){
  unsigned int height = size;
  unsigned int width = size*3; 
  float ** matrix  = malloc(sizeof(float*)*height);
  for(int i = 0; i < height; i++){
    matrix[i] = malloc(sizeof(float)*width);
  }
  
  for(int i = 0; i<height; i++){
    for(int j = 0; j<width; j++){
      matrix[i][j] = (float)1.0/(size*size);
    }
  }
  return matrix;
}

char ** pixelMat(BMP_Image * img){
  unsigned int height = img->height;
  unsigned int width = img->width*3;
  char** mat = malloc(sizeof(char*) * (height));
  for(int i = 0; i < height; i++){
    mat[i] = malloc(sizeof(char)*(width));
  }
  
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic, NUM_THREADS) collapse(2)
    for(int i=0; i < height; i++){
      for(int j=0; j< width; j++){
        mat[i][j] = img->pixel[i*width+j];
      }
    }
  }

  return mat;
}

void BMP_blur(char* open, unsigned int size){
  BMP_Image * img = BMP_open(open);
  char** out_buffer = pixelMat(img);
  float** kerSn = kernel(size);

  unsigned int height = img->height;
  unsigned int width = img->width * 3;

  int M = (size-1)/2;

  omp_set_num_threads(NUM_THREADS);
  const double startTime = omp_get_wtime();
 
  #pragma omp parallel
	{
    #pragma omp for schedule(dynamic, NUM_THREADS) collapse(1)
    for(int x=M;x<height-M;x++)
	  {
      for(int y=M;y<width-M;y++)
      {
        float sum= 0.0;
        for(int i=-M;i<=M;++i)									
        {
          for(int j=-M;j<=M;++j)
          {
            sum+=(float)kerSn[i+M][j+M]*img->pixel[(x+i)*width+(y+j)];	//matrix multiplication with kernel
          }
        }
        out_buffer[x][y]=(char)sum;
      }
	  }

    #pragma omp for schedule(dynamic, NUM_THREADS)
    for(int i = 1; i<height-1; i++){
      for(int j = 1; j<width-1; j++){
        img->pixel[i*width+j] = out_buffer[i][j];
      }
    }
  }
  char* name;
  
    char filename[] = "Blur0X.bmp";
    if(size < 10) filename[5]=size+'0';
    else{
      filename[4]= (size/10)+'0';
      filename[5]= (size%10)+'0';
    }
    name = filename; 
  

  //Guardar la imagen
  if (BMP_save(img, name) == 0)
   {
     printf("Output file invalid!\n");
     BMP_destroy(img);
     free(kerSn);
     free(out_buffer);
    //  free(buffer);
  }
  // Destroy the BMP image
  BMP_destroy(img);
  free(kerSn);
  free(out_buffer);
  const double endTime = omp_get_wtime();
  printf("%s terminado en un tiempo total de (%lf)\n",name, (endTime - startTime));
}

  
int main(){
  omp_set_num_threads(NUM_THREADS);
  const double startTime = omp_get_wtime();

  #pragma omp sections
  {
    
    #pragma omp section
    BMP_blur("f7.bmp", 11);
    #pragma omp section
    BMP_blur("f7.bmp", 13);
    #pragma omp section
    BMP_blur("f7.bmp", 15);
    #pragma omp section
    BMP_blur("f7.bmp", 17);
    #pragma omp section
    BMP_blur("f7.bmp", 19);
    #pragma omp section
    BMP_blur("f7.bmp", 21);
    #pragma omp section
    BMP_blur("f7.bmp", 23);
    #pragma omp section
    BMP_blur("f7.bmp", 25);
    #pragma omp section
    BMP_blur("f7.bmp", 27);
    #pragma omp section
    BMP_blur("f7.bmp", 29);
    #pragma omp section
    BMP_blur("f7.bmp", 31);
    #pragma omp section
    BMP_blur("f7.bmp", 33);
    #pragma omp section
    BMP_blur("f7.bmp", 35);
    #pragma omp section
    BMP_blur("f7.bmp", 37);
    #pragma omp section
    BMP_blur("f7.bmp", 39);
    #pragma omp section
    BMP_blur("f7.bmp", 41);
    #pragma omp section
    BMP_blur("f7.bmp", 43);
    #pragma omp section
    BMP_blur("f7.bmp", 45);
    #pragma omp section
    BMP_blur("f7.bmp", 47);
    #pragma omp section
    BMP_blur("f7.bmp", 49);
    #pragma omp section
    BMP_blur("f7.bmp", 51);
    #pragma omp section
    BMP_blur("f7.bmp", 53);
    #pragma omp section
    BMP_blur("f7.bmp", 55);
    #pragma omp section
    BMP_blur("f7.bmp", 57);
    #pragma omp section
    BMP_blur("f7.bmp", 59);
    #pragma omp section
    BMP_blur("f7.bmp", 61);
    #pragma omp section
    BMP_blur("f7.bmp", 63);
    #pragma omp section
    BMP_blur("f7.bmp", 65);
    #pragma omp section
    BMP_blur("f7.bmp", 67);
    #pragma omp section
    BMP_blur("f7.bmp", 69);
    #pragma omp section
    BMP_blur("f7.bmp", 71);
    #pragma omp section
    BMP_blur("f7.bmp", 73);
    #pragma omp section
    BMP_blur("f7.bmp", 75);
    #pragma omp section
    BMP_blur("f7.bmp", 77);
    #pragma omp section
    BMP_blur("f7.bmp", 79);
    #pragma omp section
    BMP_blur("f7.bmp", 81);
    #pragma omp section
    BMP_blur("f7.bmp", 83);
    #pragma omp section
    BMP_blur("f7.bmp", 85);
    #pragma omp section
    BMP_blur("f7.bmp", 87);
    #pragma omp section
    BMP_blur("f7.bmp", 89);


  }

  return 0;
}
