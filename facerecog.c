// OpenCV Sample Application: facedetect.c

// Include header files
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include <stdarg.h>

// Create memory for calculations
static CvMemStorage* storage = 0;

// Create a new Haar classifier
static CvHaarClassifierCascade* cascade = 0;
static CvHaarClassifierCascade* cascade_eyes = 0;

int still = 1;
IplImage **faceImgArr = 0;
CvMat *personNumTruthMat = 0;
int nTrainFaces = 0;
int nEigens = 0;
IplImage *pAvgTrainImg = 0;
IplImage **eigenVectArr = 0;
CvMat *eigenValMat = 0;
CvMat *projectedTrainFaceMat = 0;

void learn_eigenfaces();
void recognize_eigenfaces();
void doPCA();
void storeTrainingData_eigenfaces();
int loadTrainingData_eigenfaces(CvMat **pTrainPersonNumMat);
int findNearestNeighbor_eigenfaces(float *projectedTestFace);
int loadFaceImgArray(char *filename);



// Function prototype for detecting and drawing an object from an image
void detect_and_draw( IplImage* image );

// Create a string that contains the cascade name
const char* cascade_name =
    "haarcascade_frontalface_alt.xml";
/*    "haarcascade_profileface.xml";*/

const char* cascade_eyes_name =
    "haarcascade_eye.xml";

void cvShowManyImages(char* title, int nArgs, ...) {

    // img - Used for getting the arguments 
    IplImage *img, *tmp;

    // DispImage - the image in which input images are to be copied
    IplImage *DispImage;

    int size;
    int i;
    int m, n;
    int x, y;

    // w - Maximum number of images in a row 
    // h - Maximum number of images in a column 
    int w, h;

    // scale - How much we have to resize the image
    float scale;
    int max;

    // If the number of arguments is lesser than 0 or greater than 12
    // return without displaying 
    if(nArgs <= 0) {
        printf("Number of arguments too small....\n");
        return;
    }
    else if(nArgs > 12) {
        printf("Number of arguments too large....\n");
        return;
    }
    // Determine the size of the image, 
    // and the number of rows/cols 
    // from number of arguments 
    else if (nArgs == 1) {
        w = h = 1;
        size = 300;
    }
    else if (nArgs == 2) {
        w = 2; h = 1;
        size = 300;
    }
    else if (nArgs == 3 || nArgs == 4) {
        w = 2; h = 2;
        size = 300;
    }
    else if (nArgs == 5 || nArgs == 6) {
        w = 3; h = 2;
        size = 200;
    }
    else if (nArgs == 7 || nArgs == 8) {
        w = 4; h = 2;
        size = 200;
    }
    else {
        w = 4; h = 3;
        size = 150;
    }

    // Create a new 3 channel image
    DispImage = cvCreateImage( cvSize(100 + size*w, 60 + size*h), 8, 3 );
    cvZero(DispImage);

    // Used to get the arguments passed
    va_list args;
    va_start(args, nArgs);

    // Loop for nArgs number of arguments
    for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {

        // Get the Pointer to the IplImage
        img = va_arg(args, IplImage*);

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if(img == 0) {
            printf("Invalid arguments");
            cvReleaseImage(&DispImage);
            return;
        }

        // Find the width and height of the image
        x = img->width;
        y = img->height;

        // Find whether height or width is greater in order to resize the image
        max = (x > y)? x: y;

        // Find the scaling factor to resize the image
        scale = (float) ( (float) max / size );

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n+= 20 + size;
        }

        // Set the image ROI to display the current image
        cvSetImageROI(DispImage, cvRect(m, n, (int)( x/scale ), (int)( y/scale )));


	if (img->nChannels == 1) {
		tmp = cvCreateImage(cvSize(x, y), 8, 3);
		cvCvtColor(img, tmp, CV_GRAY2RGB);
		cvResize(tmp, DispImage, CV_INTER_LINEAR);
		cvReleaseImage(&tmp);
	} else {
        	// Resize the input image and copy the it to the Single Big Image
        	cvResize(img, DispImage, CV_INTER_LINEAR);
	}
        // Reset the ROI in order to display the next image
        cvResetImageROI(DispImage);
    }

    // Create a new window, and show the Single Big Image
    //cvNamedWindow( title, 1 );
    cvShowImage( title, DispImage);

    if (still)
    	cvWaitKey(0);
    //cvDestroyWindow(title);

    // End the number of arguments
    va_end(args);

    // Release the Image Memory
    cvReleaseImage(&DispImage);
}

int loadFaceImgArray(char *fname) {
	FILE *imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces=0;
	imgListFile = fopen(fname, "r");
	while (fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	faceImgArr = (IplImage **)cvAlloc(nFaces*sizeof(IplImage *));
	personNumTruthMat = cvCreateMat(1, nFaces, CV_32SC1);
	
	for(iFace=0;iFace<nFaces;iFace++) {
		fscanf(imgListFile, "%d %s", personNumTruthMat->data.i+iFace, imgFilename);
		faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);
	}
	fclose(imgListFile);
	return nFaces;
}

void doPCA() {
	int i;
	CvTermCriteria calcLimit;
	CvSize faceImgSize;
	nEigens = nTrainFaces-1;
	
	faceImgSize.width = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for(i=0;i<nEigens;i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);
	// alloc eigenvalue array
	eigenValMat = cvCreateMat(1, nEigens, CV_32FC1);
	// alloc average image
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);
	// set pca termination condition
	calcLimit = cvTermCriteria(CV_TERMCRIT_ITER, nEigens, 1);

	// compute avg image, eigenvalues and vectors
	cvCalcEigenObjects(nTrainFaces, (void*)faceImgArr, (void*)eigenVectArr,
			   CV_EIGOBJ_NO_CALLBACK, 0, 0, &calcLimit,
			   pAvgTrainImg, eigenValMat->data.fl);
}

void storeTrainingData_eigenfaces() {
	CvFileStorage *fileStorage;
	int i;
	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_WRITE);

	cvWriteInt(fileStorage, "nEigens", nEigens);
	cvWriteInt(fileStorage, "nTrainFaces", nTrainFaces);
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	for(i=0;i<nEigens;i++) {
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		cvWrite(fileStorage, varname, eigenVectArr[i],cvAttrList(0,0));
	}
	cvReleaseFileStorage(&fileStorage);
}
	
void learn_eigenfaces() {
	int i, offset;
	nTrainFaces = loadFaceImgArray("train_eigen.txt");
	if (nTrainFaces < 2) {
		fprintf(stderr, "Need more than 2 faces to train\n");
		return;
	}

	doPCA();
	
	projectedTrainFaceMat = cvCreateMat(nTrainFaces, nEigens, CV_32FC1);
	offset = projectedTrainFaceMat->step / sizeof(float);
	for(i=0;i<nTrainFaces;i++) {
		cvEigenDecomposite(faceImgArr[i], nEigens, eigenVectArr, 0, 0,
				   pAvgTrainImg, projectedTrainFaceMat->data.fl + i*nEigens);

	}
	storeTrainingData_eigenfaces();
}

int loadTrainingData_eigenfaces(CvMat **pTrainPersonNumMat) {
	CvFileStorage *fileStorage;
	int i;
	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_READ);
	if (!fileStorage) {
		fprintf(stderr, "Can't open facedata.xml\n");
		return 0;
	}
	
	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	for(i=0;i<nEigens;i++) {
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}
	cvReleaseFileStorage(&fileStorage);
	return 1;
}

int findNearestNeighbor_eigenfaces(float *projectedTestFace) {
	double leastDistSq = DBL_MAX;
	int i, iTrain, iNearest = 0;
	for(iTrain=0;iTrain<nTrainFaces;iTrain++) {
		double distSq = 0;
		for(i=0;i<nEigens;i++) {
			float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
			distSq += d_i*d_i;
		}

		if (distSq < leastDistSq) {
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}
	return iNearest;
}

void recognize_eigenfaces() {
	int i, nTestFaces = 0;
	CvMat *trainPersonNumMat = 0;
	float *projectedTestFace = 0;

	nTestFaces = loadFaceImgArray("test.txt");
	printf("%d test faces loaded\n", nTestFaces);
	if (!loadTrainingData_eigenfaces(&trainPersonNumMat)) return;

	projectedTestFace = (float *)cvAlloc(nEigens*sizeof(float));
	for(i=0;i<nTestFaces;i++) {
		int iNearest, nearest, truth;
		cvEigenDecomposite(faceImgArr[i], nEigens, eigenVectArr,
				   0, 0, pAvgTrainImg, projectedTestFace);

		iNearest = findNearestNeighbor_eigenfaces(projectedTestFace);
		truth = personNumTruthMat->data.i[i];
		nearest = trainPersonNumMat->data.i[iNearest];

		printf("nearest = %d, truth = %d\n", nearest, truth);
	}
}

void process_image(IplImage *img) {
	detect_and_draw(img);

}

// Main function, defines the entry point for the program.
int main( int argc, char** argv )
{
    int scale = 2;
    // Structure for getting video from camera or avi
    CvCapture* capture = 0;
    // Images to capture the frame from video or camera or from file
    IplImage *frame = 0, *frame_copy = 0;
    // Used for calculations
    int optlen = strlen("--cascade=");
    // Input file name for avi or image file.
    const char* input_name;

    // Check for the correct usage of the command line
    if( argc > 1 && strncmp( argv[1], "--cascade=", optlen ) == 0 )
    {
        cascade_name = argv[1] + optlen;
        input_name = argc > 2 ? argv[2] : 0;
    } else if (strncmp(argv[1], "train", 5) == 0) {
	learn_eigenfaces();
	exit(0);
    } else if (strncmp(argv[1], "test", 4) == 0) {
	recognize_eigenfaces();
	exit(0);
    } else {
        fprintf( stderr,
        "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" );
        return -1;
        /*input_name = argc > 1 ? argv[1] : 0;*/
    }

    // Load the HaarClassifierCascade
    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
    
    // Check whether the cascade has loaded successfully. Else report and error and quit
    if( !cascade )
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        return -1;
    }
   
    cascade_eyes = (CvHaarClassifierCascade*)cvLoad(cascade_eyes_name, 0, 0, 0 );
    if (!cascade_eyes) {
	fprintf(stderr, "ERROR: failed to load eye classifier cascade\n" );
	return -1;
    }
 
    char *ext = strrchr(input_name, '.');
    // Allocate the memory storage
    storage = cvCreateMemStorage(0);
    // Find whether to detect the object from file or from camera.
    if( !input_name || (isdigit(input_name[0]) && input_name[1] == '\0') ){
        capture = cvCaptureFromCAM( !input_name ? 0 : input_name[0] - '0' );
    } else if (ext && strncmp(ext, ".txt", 4) == 0) {
	capture = NULL;
    } else
        capture = cvCaptureFromAVI( input_name ); 

    // Create a new named window with title: result
    cvNamedWindow( "result", 1 );
    // Find if the capture is loaded successfully or not.

    // If loaded succesfully, then:
    if( capture )
    {
 
        // Capture from the camera.
        for(;;)
        {
            // Capture the frame and load it in IplImage
            if( !cvGrabFrame( capture ))
                break;
            frame = cvRetrieveFrame( capture, 0 );

            // If the frame does not exist, quit the loop
            if( !frame )
                break;

            if (!frame_copy) {
             	   printf("Allocate image\n");
		   frame_copy = cvCreateImage(cvSize(frame->width/2,frame->height/2),
                                   8, 3);
	    }
            cvResize(frame, frame_copy, CV_INTER_LINEAR);
 	    //cvCopy(frame, frame_copy,0);

            // Call the function to detect and draw the face
            //detect_and_draw( frame_copy );
	    process_image(frame_copy);
	    //cvShowImage("result", frame_copy);
            // Wait for a while before proceeding to the next frame
            cvWaitKey(1);
	    //if( cvWaitKey( 10 ) >= 0 )
            //    break;
        }

        // Release the images, and capture memory
        cvReleaseImage( &frame_copy );
	//cvReleaseImage( &frame_resized );
        cvReleaseCapture( &capture );
    }

    // If the capture is not loaded succesfully, then:
    else
    {
	still = 1;
        // Assume the image to be lena.jpg, or the input_name specified
        const char* filename = input_name ? input_name : (char*)"lena.jpg";

	IplImage* image = NULL;
	printf("%s\n", filename);
	if (strncmp(strrchr(filename, '.')+1, "txt", 3) != 0) {
        // Load the image from that filename
            image = cvLoadImage( filename, 1 );

        // If Image is loaded succesfully, then:
        //if( image )
        //{
            // Detect and draw the face
            //detect_and_draw( image );
	    process_image(image);
            // Wait for user input
            cvWaitKey(0);

            // Release the image memory
            cvReleaseImage( &image );
        }
        else
        {
	    printf("Not an image\n");
            /* assume it is a text file containing the
               list of the image filenames to be processed - one per line */
            FILE* f = fopen( filename, "rt" );
            if( f )
            {
                char buf[1000+1];

                // Get the line from the file
                while( fgets( buf, 1000, f ) )
                {

                    // Remove the spaces if any, and clean up the name
                    int len = (int)strlen(buf);
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';

                    // Load the image from the filename present in the buffer
                    image = cvLoadImage( buf, 1 );

                    // If the image was loaded succesfully, then:
                    if( image )
                    {
                        // Detect and draw the face from the image
                        //detect_and_draw( image );
                        process_image(image);
                        // Wait for the user input, and release the memory
                        cvWaitKey(0);
                        cvReleaseImage( &image );
                    }
                }
                // Close the file
                fclose(f);
            }
        }

    }
    
    // Destroy the window previously created with filename: "result"
    cvDestroyWindow("result");

    // return 0 to indicate successfull execution of the program
    return 0;
}

#define NUM_FACES 3

// Function to detect and draw any faces that is present in an image
void detect_and_draw( IplImage* temp )
{
    IplImage *grey = cvCreateImage(cvGetSize(temp), 8, 1);
    cvCvtColor(temp, grey, CV_RGB2GRAY);
    IplImage* face = cvCreateImage(cvSize(100,100), 8, 1);
    IplImage *faces_hist[NUM_FACES];
    int i,j;
    for(i=0;i<NUM_FACES;i++) {
	faces_hist[i] = cvCreateImage(cvSize(100,100), 8, 1);	
    	cvZero(faces_hist[i]);
    }

    cvZero(face);
    // Create two points to represent the face locations
    CvPoint pt1, pt2, e_pt1, e_pt2;

    // Clear the memory storage which was used before
    cvClearMemStorage( storage );

    // Find whether the cascade is loaded, to find the faces. If yes, then:
    if( cascade )
    {

        // There can be more than one face in an image. So create a growable sequence of faces.
        // Detect the objects and store them in the sequence
        CvSeq* faces = cvHaarDetectObjects( grey, cascade, storage,
                                            1.1, 2, CV_HAAR_DO_CANNY_PRUNING,
                                            cvSize(40, 40) );

        // Loop the number of faces found.
        if (faces)
		printf("Number of faces: %d\n", faces->total);
	for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
           // Create a new rectangle for drawing the face
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );

            // Find the dimensions of the face,and scale it if necessary
            pt1.x = r->x;
            pt2.x = (r->x+r->width);
            pt1.y = r->y;
            pt2.y = (r->y+r->height);

            cvSetImageROI(grey, cvRect(pt1.x, pt1.y, r->width, r->height));


            CvSeq* eyes = cvHaarDetectObjects(grey, cascade_eyes, storage, 1.1, 5, 0, cvSize(25,15));
            printf("Eyes: %p num: %d\n", eyes, eyes->total);
            for( j=0;j < (eyes ? eyes->total : 0); j++ ) {
                CvRect *e = (CvRect*)cvGetSeqElem(eyes, j);
                e_pt1.x = e->x;
                e_pt2.x = (e->x+e->width);
                e_pt1.y = e->y;
                e_pt2.y = (e->y+e->height);
                cvRectangle(grey, e_pt1, e_pt2, CV_RGB(255,255,255), 3, 8, 0);
            }

	    cvResize(grey, face, CV_INTER_LINEAR);
	    cvResetImageROI(grey);

            if (i < NUM_FACES)
		cvEqualizeHist(face, faces_hist[i]);
            
	    // Draw the rectangle in the input image
            cvRectangle( grey, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
        }
    }

    // Show the image in the window named "result"
    //cvShowImage( "result", temp );
    cvShowManyImages("result", 6, temp, grey, face, faces_hist[0], faces_hist[1], faces_hist[2]);

    // Release the temp image created.
    cvReleaseImage( &face );
    cvReleaseImage( &grey );
    for(i=0;i<NUM_FACES;i++)
	cvReleaseImage(&faces_hist[i]);
}

