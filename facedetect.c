// OpenCV Sample Application: facedetect.c

// Include header files
#include <opencv/cv.h>
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

// Function prototype for detecting and drawing an object from an image
void detect_and_draw( IplImage* image );

// Create a string that contains the cascade name
const char* cascade_name =
    "haarcascade_frontalface_alt.xml";
/*    "haarcascade_profileface.xml";*/

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

    //cvWaitKey(0);
    //cvDestroyWindow(title);

    // End the number of arguments
    va_end(args);

    // Release the Image Memory
    cvReleaseImage(&DispImage);
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
    }
    else
    {
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
    
    // Allocate the memory storage
    storage = cvCreateMemStorage(0);
    // Find whether to detect the object from file or from camera.
    if( !input_name || (isdigit(input_name[0]) && input_name[1] == '\0') ){
        capture = cvCaptureFromCAM( !input_name ? 0 : input_name[0] - '0' );
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

            // Call the function to detect and draw the face
            detect_and_draw( frame_copy );
	    //cvShowImage("result", frame_copy);
            // Wait for a while before proceeding to the next frame
            if( cvWaitKey( 10 ) >= 0 )
                break;
        }

        // Release the images, and capture memory
        cvReleaseImage( &frame_copy );
	//cvReleaseImage( &frame_resized );
        cvReleaseCapture( &capture );
    }

    // If the capture is not loaded succesfully, then:
    else
    {
        // Assume the image to be lena.jpg, or the input_name specified
        const char* filename = input_name ? input_name : (char*)"lena.jpg";

        // Load the image from that filename
        IplImage* image = cvLoadImage( filename, 1 );

        // If Image is loaded succesfully, then:
        if( image )
        {
            // Detect and draw the face
            detect_and_draw( image );

            // Wait for user input
            cvWaitKey(0);

            // Release the image memory
            cvReleaseImage( &image );
        }
        else
        {
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
                        detect_and_draw( image );
                        
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

// Function to detect and draw any faces that is present in an image
void detect_and_draw( IplImage* temp )
{
//    int scale = 2;

    // Create a new image based on the input image
//    IplImage* temp = cvCreateImage( cvSize(img->width/scale,img->height/scale), 8, 3 );
//    cvResize(img, temp, CV_INTER_LINEAR);
    IplImage *grey = cvCreateImage(cvGetSize(temp), 8, 1);
    cvCvtColor(temp, grey, CV_RGB2GRAY);
    IplImage* face = cvCreateImage(cvSize(40,40), 8, 1);
    IplImage* face_hist = cvCreateImage(cvSize(40,40), 8, 1);
    cvZero(face);
    cvZero(face_hist);
    // Create two points to represent the face locations
    CvPoint pt1, pt2;
    int i;

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
            cvResize(grey, face, CV_INTER_LINEAR);
            cvResetImageROI(grey);
	    cvEqualizeHist(face,face_hist);
            // Draw the rectangle in the input image
            cvRectangle( grey, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
        }
    }

    // Show the image in the window named "result"
    //cvShowImage( "result", temp );
    cvShowManyImages("result", 6, temp, grey, face, face_hist, temp, temp);

    // Release the temp image created.
    cvReleaseImage( &face );
    cvReleaseImage( &grey );
}

