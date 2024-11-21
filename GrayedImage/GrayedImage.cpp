#include <windows.h>
#include <gdiplus.h>
#include <commdlg.h>
#include <chrono>
#include <vector>
#include <string>
#include <immintrin.h>

#pragma comment(lib, "gdiplus.lib")

using namespace Gdiplus;

HINSTANCE hInst;
HWND hwndDefault, hwndScalar, hwndSimd, hwndTimeScalar, hwndTimeSimd, hwndBtnOpenFile;
WCHAR filePath[MAX_PATH] = L"";

CONST INT maxWidth = 400;
CONST INT maxHeight = 600;

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void ScalarGray(Bitmap* input, Bitmap* output);
void SIMDGray(Bitmap* input, Bitmap* output);
void OnOpenFileClick(HWND hwnd);
void ResizeAndDisplayImage(HWND hwnd, Bitmap* bmp);


int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nShowCmd)
{
	GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

	WNDCLASS wc = { 0 };
	wc.lpfnWndProc = WndProc;
	wc.hInstance = hInstance;
	wc.lpszClassName = L"ImageProcessingApp";

	RegisterClass(&wc);

	hInst = hInstance;

	HWND hwnd = CreateWindow(L"ImageProcessingApp", L"Image Processing App", WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, 1600, 800, NULL, NULL, hInstance, NULL);

	hwndBtnOpenFile = CreateWindow(L"BUTTON", L"Open Image", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
		300, 10, 200, 30, hwnd, (HMENU)1, hInst, NULL);

	ShowWindow(hwnd, nShowCmd);
	UpdateWindow(hwnd);

	MSG msg;
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	GdiplusShutdown(gdiplusToken);
	return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_CREATE:
		hwndDefault = CreateWindow(L"STATIC", L"Default Image", WS_VISIBLE | WS_CHILD | SS_BITMAP,
			50, 50, maxWidth, maxHeight, hwnd, NULL, hInst, NULL);
		hwndScalar = CreateWindow(L"STATIC", L"Scalar Image", WS_VISIBLE | WS_CHILD | SS_BITMAP,
			550, 50, maxWidth, maxHeight, hwnd, NULL, hInst, NULL);
		hwndSimd = CreateWindow(L"STATIC", L"SIMD Image", WS_VISIBLE | WS_CHILD | SS_BITMAP,
			1100, 50, maxWidth, maxHeight, hwnd, NULL, hInst, NULL);
		hwndTimeScalar = CreateWindow(L"STATIC", L"Time: N/A", WS_VISIBLE | WS_CHILD,
			700, 700, 200, 20, hwnd, NULL, hInst, NULL);
		hwndTimeSimd = CreateWindow(L"STATIC", L"Time: N/A", WS_VISIBLE | WS_CHILD,
			1300, 700, 200, 20, hwnd, NULL, hInst, NULL);
		break;

	case WM_COMMAND:
		if (LOWORD(wParam) == 1)
			OnOpenFileClick(hwnd);
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	default:
		return DefWindowProc(hwnd, message, wParam, lParam);
	}
	return 0;
}

void OnOpenFileClick(HWND hwnd)
{
	OPENFILENAME ofn = { 0 };
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFilter = L"Image Files\0*.png;*.jpg;*.jpeg\0All Files\0*.*\0";
	ofn.lpstrFile = filePath;
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

	if (GetOpenFileName(&ofn))
	{
		Bitmap* bmp = new Bitmap(filePath);
		if (bmp->GetLastStatus() != Ok)
		{
			MessageBox(hwnd, L"Failed to load image.", L"Error", MB_ICONERROR);
			delete bmp;
			return;
		}

		SendMessage(hwndDefault, STM_SETIMAGE, IMAGE_BITMAP, 0);
		SendMessage(hwndScalar, STM_SETIMAGE, IMAGE_BITMAP, 0);
		SendMessage(hwndSimd, STM_SETIMAGE, IMAGE_BITMAP, 0);

		ResizeAndDisplayImage(hwndDefault, bmp);

		Bitmap* scalarBmp = new Bitmap(bmp->GetWidth(), bmp->GetHeight(), PixelFormat32bppARGB);
		auto start = std::chrono::high_resolution_clock::now();
		ScalarGray(bmp, scalarBmp);
		auto end = std::chrono::high_resolution_clock::now();
		SetWindowText(hwndTimeScalar, (L"Time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) + L"ms").c_str());
		ResizeAndDisplayImage(hwndScalar, scalarBmp);
		delete scalarBmp;

		Bitmap* simdBmp = new Bitmap(bmp->GetWidth(), bmp->GetHeight(), PixelFormat32bppARGB);
		start = std::chrono::high_resolution_clock::now();
		SIMDGray(bmp, simdBmp);
		end = std::chrono::high_resolution_clock::now();
		SetWindowText(hwndTimeSimd, (L"Time: " + std::to_wstring(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) + L"ms").c_str());
		ResizeAndDisplayImage(hwndSimd, simdBmp);
		delete simdBmp;

		delete bmp;
	}
}

void ResizeAndDisplayImage(HWND hwnd, Bitmap* bmp)
{
	HBITMAP oldBitmap = (HBITMAP)SendMessage(hwnd, STM_GETIMAGE, IMAGE_BITMAP, 0);
	
	if (oldBitmap)
		DeleteObject(oldBitmap);

	Bitmap* whiteBmp = new Bitmap(maxWidth, maxHeight, PixelFormat32bppARGB);
	Graphics graphics(whiteBmp);
	graphics.Clear(Color::White);

	HBITMAP hWhiteBitmap;
	whiteBmp->GetHBITMAP(Color::White, &hWhiteBitmap);

	SendMessage(hwnd, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)hWhiteBitmap);

	delete whiteBmp;

	INT width = bmp->GetWidth();
	INT height = bmp->GetHeight();

	FLOAT aspectRatioWidth = width / static_cast<FLOAT>(maxWidth);
	FLOAT aspectRatioHeight = height / static_cast<FLOAT>(maxHeight);

	if (aspectRatioWidth > 1.0f || aspectRatioHeight > 1.0f)
	{
		FLOAT scaleFactor = max(aspectRatioWidth, aspectRatioHeight);
		width = static_cast<INT>(width / scaleFactor);
		height = static_cast<INT>(height / scaleFactor);
	}

	Bitmap* resizedBmp = new Bitmap(width, height, PixelFormat32bppARGB);
	Graphics resizedGraphics(resizedBmp);
	resizedGraphics.DrawImage(bmp, 0, 0, width, height);

	HBITMAP hBitmap;
	resizedBmp->GetHBITMAP(Color::White, &hBitmap);

	SendMessage(hwnd, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)hBitmap);

	delete resizedBmp;

	InvalidateRect(hwnd, NULL, TRUE);
}

void ScalarGray(Bitmap* input, Bitmap* output)
{
	UINT width = input->GetWidth();
	UINT height = input->GetHeight();

	for (UINT y = 0; y < height; ++y)
	{
		BitmapData inputData, outputData;
		Rect rect(0, y, width, 1);

		input->LockBits(&rect, ImageLockModeRead, PixelFormat32bppARGB, &inputData);
		output->LockBits(&rect, ImageLockModeWrite, PixelFormat32bppARGB, &outputData);

		BYTE* inputRow = (BYTE*)inputData.Scan0;
		BYTE* outputRow = (BYTE*)outputData.Scan0;

		for (UINT x = 0; x < width; ++x)
		{
			BYTE b = inputRow[x * 4];
			BYTE g = inputRow[x * 4 + 1];
			BYTE r = inputRow[x * 4 + 2];
			BYTE gray = static_cast<BYTE>(0.299 * r + 0.587 * g + 0.114 * b);
			outputRow[x * 4] = outputRow[x * 4 + 1] = outputRow[x * 4 + 2] = gray;
			outputRow[x * 4 + 3] = inputRow[x * 4 + 3];
		}

		input->UnlockBits(&inputData);
		output->UnlockBits(&outputData);
	}
}

void SIMDGray(Bitmap* input, Bitmap* output)
{
	UINT width = input->GetWidth();
	UINT height = input->GetHeight();

	__m256 grayCoeffR = _mm256_set1_ps(0.299f);
	__m256 grayCoeffG = _mm256_set1_ps(0.587f);
	__m256 grayCoeffB = _mm256_set1_ps(0.114f);
	__m256i bitmap = _mm256_set1_epi32(0x000000FF);

	for (UINT y = 0; y < height; ++y)
	{
		BitmapData inputData, outputData;
		Rect rect(0, y, width, 1);

		input->LockBits(&rect, ImageLockModeRead, PixelFormat32bppARGB, &inputData);
		output->LockBits(&rect, ImageLockModeWrite, PixelFormat32bppARGB, &outputData);

		BYTE* inputRow = (BYTE*)inputData.Scan0;
		BYTE* outputRow = (BYTE*)outputData.Scan0;

		UINT x = 0;
		UINT alignedWidth = width / 8 * 8;

		for (; x < alignedWidth; x += 8)
		{
			__m256i pixelData = _mm256_loadu_si256((__m256i*)(inputRow + x * 4));

			__m256i b = _mm256_and_si256(pixelData, bitmap);
			__m256i g = _mm256_and_si256(_mm256_srli_epi32(pixelData, 8), bitmap);
			__m256i r = _mm256_and_si256(_mm256_srli_epi32(pixelData, 16), bitmap);
			__m256i a = _mm256_and_si256(_mm256_srli_epi32(pixelData, 24), bitmap);

			__m256 b_f = _mm256_cvtepi32_ps(b);
			__m256 g_f = _mm256_cvtepi32_ps(g);
			__m256 r_f = _mm256_cvtepi32_ps(r);

			__m256 gray_f = _mm256_add_ps(
				_mm256_add_ps(_mm256_mul_ps(r_f, grayCoeffR), _mm256_mul_ps(g_f, grayCoeffG)),
				_mm256_mul_ps(b_f, grayCoeffB));

			__m256i gray = _mm256_cvtps_epi32(gray_f);
			gray = _mm256_min_epi32(gray, bitmap);

			__m256i grayPixel = _mm256_or_si256(
				_mm256_or_si256(gray, _mm256_slli_epi32(gray, 8)),
				_mm256_or_si256(_mm256_slli_epi32(gray, 16), _mm256_slli_epi32(a, 24)));

			_mm256_storeu_si256((__m256i*)(outputRow + x * 4), grayPixel);
		}

		for (; x < width; ++x)
		{
			BYTE b = inputRow[x * 4];
			BYTE g = inputRow[x * 4 + 1];
			BYTE r = inputRow[x * 4 + 2];
			BYTE gray = static_cast<BYTE>(0.299 * r + 0.587 * g + 0.114 * b);
			outputRow[x * 4] = outputRow[x * 4 + 1] = outputRow[x * 4 + 2] = gray;
			outputRow[x * 4 + 3] = inputRow[x * 4 + 3];
		}

		input->UnlockBits(&inputData);
		output->UnlockBits(&outputData);
	}
}
