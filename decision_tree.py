print("======= Pohon Keputusan =======\n")

print("Rata_Rata_SMA <= 75")
print("  True:")
print("    [Label: Tidak Diterima]")
print("  False:")
print("    Jurusan_SMA == IPS")
print("      True:")
print("        [Label: Sistem Informasi]")
print("      False:")
print("        Nilai_UTBK <= 75")
print("          True:")
print("            [Label: RPL]")
print("          False:")
print("            Nilai_UTBK <= 85")
print("              True:")
print("                [Label: Sains Data]")
print("              False:")
print("                [Label: Informatika]")

print("\n----- Masukkan data untuk prediksi: -----\n")

jurusan_sma = input("Jurusan SMA (IPA/IPS)             : ")
rata_rata_sma = int(input("Rata-rata Nilai SMA (0-100)       : "))
nilai_utbk = int(input("Nilai UTBK (0-100)                : "))

# ===== Logika Keputusan =====
if rata_rata_sma <= 75:
    hasil = "Tidak Diterima"
else:
    if jurusan_sma.upper() == "IPS":
        hasil = "Sistem Informasi"
    else:  # IPA
        if nilai_utbk <= 75:
            hasil = "RPL"
        elif nilai_utbk <= 85:
            hasil = "Sains Data"
        else:
            hasil = "Informatika"

print("\nHasil Prediksi                    :", hasil)
