from typing import *
import pytest
from unittest.mock import patch

import os
import urllib.request
from lumiress.io.downloader import(
    WeightDownloader,
    _weight_urls,
    _base_path,
    _base_url,
    _weight_version,
    _download_weight,
    download_contrast_enhancement,
    download_super_resolution,
    download_real_denoising,
    download_lowlight_enhancement,
    download_all
)

def test_weight_downloader_class(mocker):
    # Mock out the dependencies
    mock_path = mocker.MagicMock()
    mock_path.exists.return_value = False

    # Mock the hash computation function
    mocker.patch("lumiress.io.downloader.WeightDownloader._calculate_sha256", return_value="dummyhash")

    # Instantiate the class with the mock object
    wd = WeightDownloader(mock_path, 'http://example.com', 'dummyhash', verbose=False)
    wd.weight_base_path = mock_path
    wd.weight_path = mock_path

    # Mock the urlretrieve method
    mock_urlretrieve = mocker.patch("urllib.request.urlretrieve")

    # Mock the os.remove method
    mock_os_remove = mocker.patch("os.remove")

    # Mock the os.makedirs method
    mock_os_makedirs = mocker.patch("os.makedirs")

    # Call the method under test
    wd.check_and_download_weights()

    # Assert that the mocks have been called as expected
    mock_os_makedirs.assert_called()
    mock_urlretrieve.assert_called_with('http://example.com', mock_path, reporthook=wd._reporthook)
    

def test_download_weight(mocker):
    # Given
    name = "real_denoising"
    expected_url = _weight_urls[_weight_version][name]["url"]
    expected_sha256sum = _weight_urls[_weight_version][name]["sha256sum"]

    # Mock the WeightDownloader's instance and class
    mock_wd_instance = mocker.MagicMock()
    mock_wd_class = mocker.patch('lumiress.io.downloader.WeightDownloader', return_value=mock_wd_instance)

    # When
    _download_weight(name)

    # Then
    mock_wd_class.assert_called_once_with(_base_path, expected_url, expected_sha256sum, verbose=True)
    mock_wd_instance.check_and_download_weights.assert_called_once()


def test_download_weight_nonexistent(mocker):
    # Given a non-existent weight name
    name = "non_existent"

    # When/Then
    with pytest.raises(ValueError) as e_info:
        _download_weight(name)

    # Check if ValueError contains correct message
    assert str(e_info.value) == f"Weight {name} not found"


def test_download_real_denoising(mocker):
    mock_download_weight = mocker.patch('lumiress.io.downloader._download_weight')
    download_real_denoising()
    mock_download_weight.assert_called_once_with("real_denoising")

def test_download_super_resolution(mocker):
    mock_download_weight = mocker.patch('lumiress.io.downloader._download_weight')
    download_super_resolution()
    mock_download_weight.assert_called_once_with("super_resolution")

def test_download_contrast_enhancement(mocker):
    mock_download_weight = mocker.patch('lumiress.io.downloader._download_weight')
    download_contrast_enhancement()
    mock_download_weight.assert_called_once_with("contrast_enhancement")

def test_download_lowlight_enhancement(mocker):
    mock_download_weight = mocker.patch('lumiress.io.downloader._download_weight')
    download_lowlight_enhancement()
    mock_download_weight.assert_called_once_with("lowlight_enhancement")

def test_download_all(mocker):
    mock_download_real_denoising = mocker.patch('lumiress.io.downloader.download_real_denoising')
    mock_download_super_resolution = mocker.patch('lumiress.io.downloader.download_super_resolution')
    mock_download_contrast_enhancement = mocker.patch('lumiress.io.downloader.download_contrast_enhancement')
    mock_download_lowlight_enhancement = mocker.patch('lumiress.io.downloader.download_lowlight_enhancement')

    download_all()

    mock_download_real_denoising.assert_called_once()
    mock_download_super_resolution.assert_called_once()
    mock_download_contrast_enhancement.assert_called_once()
    mock_download_lowlight_enhancement.assert_called_once()
