# Hermes Agent for Android

Hermes Agent is now available as a native Android APK, making it easy to run the world's most advanced AI agent on your mobile device.

## Quick Installation

### Option 1: Download and Install APK

The easiest way to get Hermes Agent on your Android device:

1. **Download the APK** from our releases:
   
   [![Download Latest APK](https://img.shields.io/badge/Download-APK-green?style=for-the-badge)](https://github.com/NousResearch/hermes-agent/releases/latest/download/Hermes-Agent-latest.apk)

2. **Enable unknown sources** on your Android device:
   - Open **Settings**
   - Go to **Security** or **Privacy**
   - Enable **Install unknown apps** or **Install from unknown sources**
   - Select your browser or file manager

3. **Open the downloaded APK file** and tap **Install**

4. **Launch Hermes Agent** from your app drawer

That's it! Hermes Agent will download any necessary dependencies on first launch.

### Option 2: Using Termux (Advanced Users)

For power users who prefer the terminal experience:

```bash
# In Termux:
pkg update && pkg upgrade
pkg install wget
wget https://github.com/NousResearch/hermes-agent/releases/latest/download/hermes-agent_<version>_aarch64.deb
dpkg -i hermes-agent_<version>_aarch64.deb
hermes
```

## Features

✅ **Full AI Agent Experience** - Run the same powerful agent as desktop
✅ **Continuous Learning** - All your skills and memories persist
✅ **Multi-Model Support** - Use any model provider (OpenRouter, Nous, etc.)
✅ **Offline Capable** - Works without internet connection for local models
✅ **Background Execution** - Keep your agent running in the background
✅ **Notifications** - Get notified of important events

## Requirements

- **Android 7.0 (Nougat) or later**
- **Minimum 2GB RAM** recommended
- **500MB+ free storage** for dependencies
- **Internet connection** for first setup and model downloads

## First Launch

When you first launch Hermes Agent:

1. The app will request necessary permissions
2. It will download and install Python runtime (if not already present)
3. It will download the latest Hermes Agent package
4. You'll be taken to the main interface

This may take a few minutes depending on your internet speed.

## Usage

### Starting a Conversation

Simply type your message and press Enter. Hermes will:
- Understand your request
- Use appropriate tools if needed
- Maintain context throughout the conversation
- Learn from the interaction

### Switching Models

To change the AI model:

```
hermes model
```

This will show you available models and allow you to select one.

### Managing Skills

Hermes Agent learns and improves over time. Your custom skills are stored locally on your device.

## Configuration

### Environment Variables

You can configure Hermes Agent by creating a `.env` file in the app's data directory:

```bash
# Model provider
MODEL_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key

# Or use Nous Research
MODEL_PROVIDER=nous
NOUS_API_KEY=your_api_key
```

### Storage Location

All data is stored in:
```
/data/data/com.nousresearch.hermes/files/hermes/
```

Includes:
- Conversation history
- Skills and memories
- Configuration files
- Downloaded models (if any)

## Troubleshooting

### Installation Issues

**"App not installed" error:**
- Make sure you have enough storage space
- Try uninstalling any previous version first
- Check that you've enabled "Install unknown apps" for your browser

**"Parse error" when installing:**
- The APK may be corrupted. Download it again.
- Make sure you're downloading the correct version for your device architecture (arm64)

### Runtime Issues

**App crashes on startup:**
- Try clearing the app data (Settings > Apps > Hermes Agent > Storage > Clear Data)
- Reinstall the APK
- Check the log files in `/data/data/com.nousresearch.hermes/files/logs/`

**Python initialization fails:**
- Make sure you have an internet connection
- Wait a few minutes and try again
- The first launch downloads necessary dependencies

### Performance Issues

**App is slow:**
- Close other apps to free up memory
- Use a lighter model (e.g., `mistralai/mistral-tiny`)
- Reduce the context window size in configuration

**App crashes with out of memory:**
- Try using smaller models
- Close background apps
- Consider using Termux version which may have better memory management

## Building from Source

If you want to build the APK yourself:

### Prerequisites

- Android Studio or Android SDK
- JDK 17 or later
- Android NDK (for native extensions)
- Internet connection

### Build Steps

```bash
# Clone the repository
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Run the build script
./scripts/build_android_apk.sh --version 1.0.0 --debug

# The APK will be created at:
# Hermes-Agent-1.0.0-debug.apk
```

### Release Build

For a production release:

```bash
./scripts/build_android_apk.sh --version 1.0.0 --release
```

Note: Release builds require a signing keystore. Set up your keystore in `android/app/keystore/release.keystore`.

## Contributing

We welcome contributions to improve the Android experience! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

### Testing

When testing Android-specific changes:

1. Build the APK as described above
2. Install on a physical device (emulators may not work well)
3. Test on multiple Android versions (7.0+)
4. Test with different screen sizes and densities

### Reporting Issues

When reporting Android-specific issues, please include:

- Android version
- Device model
- APK version
- Steps to reproduce
- Log output (if available)

## Security

⚠️ **Important Security Notes:**

- Hermes Agent has access to your conversations and can execute code
- Only install APKs from trusted sources (official releases)
- Review permissions carefully before installing
- The app requests storage permissions to save conversation data
- Network permissions are required to access AI models

## Privacy

Your data is your data. Hermes Agent:
- Stores all conversation data locally on your device
- Does not send your data to third parties without explicit permission
- All model interactions go through your configured provider
- You can export and delete your data at any time

## License

Hermes Agent is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.

---

## Support

For help and support:

- **GitHub Issues**: [https://github.com/NousResearch/hermes-agent/issues](https://github.com/NousResearch/hermes-agent/issues)
- **Discord**: Join our [Discord server](https://discord.gg/NousResearch)
- **Documentation**: [https://hermes-agent.nousresearch.com/docs/](https://hermes-agent.nousresearch.com/docs/)

## Versions

| Version | Release Date | Notes |
|---------|--------------|-------|
| 1.0.0 | 2024 | Initial Android APK release |
