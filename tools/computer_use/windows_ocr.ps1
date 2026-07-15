param(
  [Parameter(Mandatory=$true)]
  [string]$Path,
  [string]$Language = ""
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime]
[void][Windows.Storage.FileAccessMode, Windows.Storage, ContentType=WindowsRuntime]
[void][Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime]
[void][Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
[void][Windows.Globalization.Language, Windows.Foundation, ContentType=WindowsRuntime]

function Await-WinRt {
  param(
    [Parameter(Mandatory=$true)] $Operation,
    [Parameter(Mandatory=$true)] [type]$ResultType
  )

  $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
    Where-Object {
      $_.Name -eq "AsTask" -and
      $_.GetParameters().Count -eq 1 -and
      $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
    } |
    Select-Object -First 1

  $task = $method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
  $task.Wait()
  $task.Result
}

$resolved = (Resolve-Path -LiteralPath $Path).Path
$file = Await-WinRt ([Windows.Storage.StorageFile]::GetFileFromPathAsync($resolved)) ([Windows.Storage.StorageFile])
$stream = Await-WinRt ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await-WinRt ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await-WinRt ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

if ($bitmap.PixelWidth -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension -or
    $bitmap.PixelHeight -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension) {
  throw "Image exceeds Windows OCR maximum dimension."
}

if ($Language.Trim()) {
  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new($Language))
} else {
  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
}

if ($null -eq $engine) {
  throw "No Windows OCR engine is available for the requested language."
}

$result = Await-WinRt ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$lines = @($result.Lines | ForEach-Object { $_.Text })
$words = @(
  $result.Lines | ForEach-Object {
    $_.Words | ForEach-Object {
      [pscustomobject]@{
        text = $_.Text
        bounds = @(
          [math]::Round($_.BoundingRect.X),
          [math]::Round($_.BoundingRect.Y),
          [math]::Round($_.BoundingRect.Width),
          [math]::Round($_.BoundingRect.Height)
        )
      }
    }
  }
)

[pscustomobject]@{
  method = "windows-media-ocr"
  language = $engine.RecognizerLanguage.LanguageTag
  width = $bitmap.PixelWidth
  height = $bitmap.PixelHeight
  text = $result.Text
  lines = $lines
  words = $words
  confidence_available = $false
} | ConvertTo-Json -Depth 6 -Compress
