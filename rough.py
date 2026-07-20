try:
    from jugaad_data.nse import NSELive
    print("[OK] jugaad-data installed successfully")
except ImportError as e:
    print("[ERROR] jugaad-data is not installed")
    print(e)

print("\nAll imports completed.")