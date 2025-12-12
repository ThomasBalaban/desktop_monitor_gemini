import sounddevice as sd

def list_devices():
    print("\n🎧 Audio Device Scanner")
    print("=======================")
    print("Looking for devices...")
    
    try:
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        
        print(f"\nFound {len(devices)} devices:\n")
        
        for i, device in enumerate(devices):
            name = device['name']
            inputs = device['max_input_channels']
            outputs = device['max_output_channels']
            sample_rate = device['default_samplerate']
            
            # Filter for likely candidates
            mark = " "
            if inputs > 0:
                if "BlackHole" in name or "Multi-Output" in name:
                    mark = "✅"  # Likely what you want for Desktop Audio
                elif "Microphone" in name or "Input" in name:
                    mark = "🎤"  # Likely a Microphone
            
            print(f"{mark} [{i}] {name}")
            print(f"      In: {inputs} | Out: {outputs} | Rate: {int(sample_rate)}Hz")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Error querying devices: {e}")
        print("\nTroubleshooting for macOS:")
        print("1. Ensure you have 'portaudio' installed: brew install portaudio")
        print("2. Ensure Terminal/VSCode has Microphone permissions in System Settings.")

if __name__ == "__main__":
    list_devices()