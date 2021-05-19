import nlpaug.augmenter.word as naw

text = """
Eversport GmbH
Adresse:	
Heiligenstädterstraße 31, Stiege 2, 5. Stock, Büro links,
1190 Wien, Österreich
Telefon:	+43 660 7005041
E-Mail:	office@eversports.com
Geschäftsführer:	Hanno Lippitsch
Firmenbuchnummer:	404544v
Firmenbuchgericht:	
Handelsgericht Wien, Marxergasse 1a,
1030 Wien, Österreich
UID:	ATU68292377
DVR-Nr.:	4015117
"""

if __name__ == '__main__':
    aug = naw.ContextualWordEmbsAug(
    model_path='/mnt/data/users/keller/xlm-roberta-da', action="insert", device='cpu')
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text)