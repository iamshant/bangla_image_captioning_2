import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open("test_examples/child.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open("test_examples/boat.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open("test_examples/horse.png").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    test_img6 = transform(Image.open("test_examples/6.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 6 CORRECT: রাস্তার পাশের দোকানে একজন লোক অনেকগুলো মাঠা তৈরি করছে বিক্রি করার জন্য এবং দোকানের বাইরে কালো টি-শার্ট পরে একজন দাঁড়িয়ে লোক আছে ও রাস্তা দিয়ে কিছু মানুষ চলাচল করছে।")
    print(
        "Example 6 OUTPUT: "
        + " ".join(model.caption_image(test_img6.to(device), dataset.vocab))
    )

    test_img7 = transform(Image.open("test_examples/7.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 7 CORRECT: দুইটি একই রঙের ভবনের সামনে কয়েকটি গাছপালা রয়েছে এবং গাছপালাগুলো সামনে ফাঁকা রাস্তার মধ্য দিয়ে একজন মহিলা হেটে যাচ্ছে ও রাস্তার মাঝে রাস্তা বিভাজক রয়েছে।")
    print(
        "Example 7 OUTPUT: "
        + " ".join(model.caption_image(test_img7.to(device), dataset.vocab))
    )

    test_img8 = transform(Image.open("test_examples/8.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 8 CORRECT: প্লেনের মধ্যে অনেকগুলো কালো পোশাক পরা বাংলাদেশ সশস্ত্র বাহিনীর কয়েকজন সদস্য বসে রয়েছে ও আরো সামনের আসনে এক পাশে দুইজন মহিলা বসে রয়েছে আরেক পাশে বাংলাদেশ সশস্ত্র বাহিনীর একজন সদস্যের সাথে আরও একজন মহিলা বসে রয়েছে।")
    print(
        "Example 8 OUTPUT: "
        + " ".join(model.caption_image(test_img8.to(device), dataset.vocab))
    )

    test_img9 = transform(Image.open("test_examples/9.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 9 CORRECT: অনেকগুলো পাহাড়ের সামনে একজন মেয়ে হাত বাড়িয়ে দাঁড়িয়ে রয়েছে।")
    print(
        "Example 9 OUTPUT: "
        + " ".join(model.caption_image(test_img9.to(device), dataset.vocab))
    )

    test_img10 = transform(Image.open("test_examples/10.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 10 CORRECT: অনেকগুলো ভবন গাছপালা ও একটি স্টেডিয়ামের সামনে বড় একটি দিঘি রয়েছে।")
    print(
        "Example 10 OUTPUT: "
        + " ".join(model.caption_image(test_img10.to(device), dataset.vocab))
    )

    test_img11 = transform(Image.open("test_examples/11.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 11 CORRECT: রাস্তার পাশে কয়েকটি দোকানে কয়েকজন লোক অবস্থান নিয়েছে বৃষ্টি থেকে বাঁচার জন্য এবং রাস্তার পাশের দোকানের সামনে একটি রিক্সা রয়েছে।")
    print(
        "Example 11 OUTPUT: "
        + " ".join(model.caption_image(test_img11.to(device), dataset.vocab))
    )

    test_img12 = transform(Image.open("test_examples/12.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 12 CORRECT: দুইটি ভবনের মাঝে অনেকগুলো লোক দাঁড়িয়ে রয়েছে এবং মাটিতে অনেকগুলো মাছ ফেলে রাখা হয়েছে ও কিছু মানুষ কিছু মাছ সংগ্রহ করে কয়েকটি পাত্রে রাখছে।")
    print(
        "Example 12 OUTPUT: "
        + " ".join(model.caption_image(test_img12.to(device), dataset.vocab))
    )

    test_img13 = transform(Image.open("test_examples/13.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 13 CORRECT: রাস্তার পাশে ফুটপাত দিয়ে এবং ভারতের বাইরে দিয়ে অনেকগুলো মানুষ হেঁটে মাস্ক পরে আসছে এবং রাস্তা দিয়ে অনেকগুলো ট্রাক-প্রাইভেটকার ও রিকশা চলাচল করছে।")
    print(
        "Example 13 OUTPUT: "
        + " ".join(model.caption_image(test_img13.to(device), dataset.vocab))
    )

    test_img14 = transform(Image.open("test_examples/14.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 14 CORRECT: শহরের নামী দামী একটি রেস্টুরেন্টের ভেতরে অনেকগুলো ফাঁকা টেবিল সজ্জিত অবস্থায় রাখা আছে এবং রেস্টুরেন্টের কয়েকজন কর্মকর্তা টেবিলের সামনে দাঁড়িয়ে রয়েছে।")
    print(
        "Example 14 OUTPUT: "
        + " ".join(model.caption_image(test_img14.to(device), dataset.vocab))
    )

    test_img15 = transform(Image.open("test_examples/15.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 15 CORRECT: রাস্তার পাশে একটি দোকানের সামনে কয়েকটি সাইকেল রাখা আছে।")
    print(
        "Example 15 OUTPUT: "
        + " ".join(model.caption_image(test_img15.to(device), dataset.vocab))
    )

    test_img16 = transform(Image.open("test_examples/16.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 16 CORRECT: শাড়ী পরিহিত দুইজন মেয়ে সজ্জিত হয়ে মাথায় ফুলের মালা দিয়ে দাঁড়িয়ে রয়েছে।")
    print(
        "Example 16 OUTPUT: "
        + " ".join(model.caption_image(test_img16.to(device), dataset.vocab))
    )


    test_img17 = transform(Image.open("test_examples/17.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 17 CORRECT: বাসস্ট্যান্ডে কয়েকটি বাস দাঁড়িয়ে আছে এবং পেছনে গাছপালা আছে।")
    print(
        "Example 17 OUTPUT: "
        + " ".join(model.caption_image(test_img17.to(device), dataset.vocab))
    )

    test_img18 = transform(Image.open("test_examples/18.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 18 CORRECT: নদীতে নৌকায় তিনজন মানুষ বাস করে।")
    print(
        "Example 18 OUTPUT: "
        + " ".join(model.caption_image(test_img18.to(device), dataset.vocab))
    )

    test_img19 = transform(Image.open("test_examples/19.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 19 CORRECT: রাস্তার পাশে কয়েকটি বহুতল ভবন রয়েছে এবং ভবনে সাইনবোর্ড আছে ও রাস্তার মাঝে গাছের চারা লাগানো হয়েছে।")
    print(
        "Example 19 OUTPUT: "
        + " ".join(model.caption_image(test_img19.to(device), dataset.vocab))
    )

    test_img20 = transform(Image.open("test_examples/20.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 20 CORRECT: রেস্টুরেন্টের একটি কক্ষের মধ্যে অনেকগুলো চেয়ার টেবিল সাজিয়ে রাখা আছে।")
    print(
        "Example 20 OUTPUT: "
        + " ".join(model.caption_image(test_img20.to(device), dataset.vocab))
    )

    test_img21 = transform(Image.open("test_examples/21.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 21 CORRECT: রাস্তা দিয়ে যানবাহন চলাচল করছে এবং রাস্তার মধ্যে বাংলাদেশ সেনাবাহিনীর অনেক জন সদস্য রাইফেল কাঁধে দাঁড়িয়ে রয়েছে।")
    print(
        "Example 21 OUTPUT: "
        + " ".join(model.caption_image(test_img21.to(device), dataset.vocab))
    )

    test_img22 = transform(Image.open("test_examples/22.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 22 CORRECT: একজন পুলিশ সদস্য পুলিশের গাড়ির সামনে দাঁড়িয়ে আছে।")
    print(
        "Example 22 OUTPUT: "
        + " ".join(model.caption_image(test_img22.to(device), dataset.vocab))
    )

    test_img23 = transform(Image.open("test_examples/23.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 23 CORRECT: দুজন বিজিবির সদস্য নদীর পাড়ে দাঁড়িয়ে রয়েছে।")
    print(
        "Example 23 OUTPUT: "
        + " ".join(model.caption_image(test_img23.to(device), dataset.vocab))
    )

    test_img24 = transform(Image.open("test_examples/24.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 24 CORRECT: রানওয়ে থেকে একটি বিমান উড্ডয়ন করছে এবং পাশে ভবন ও গাছপালা রয়েছে।")
    print(
        "Example 24 OUTPUT: "
        + " ".join(model.caption_image(test_img24.to(device), dataset.vocab))
    )

    test_img25 = transform(Image.open("test_examples/25.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 25 CORRECT: একটি প্রধান ফটকের পেছনে গাছপালা ও মাঠ রয়েছে।")
    print(
        "Example 25 OUTPUT: "
        + " ".join(model.caption_image(test_img25.to(device), dataset.vocab))
    )

    test_img26 = transform(Image.open("test_examples/26.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 26 CORRECT: রাস্তা দিয়ে যানবাহন চলাচল করছে এবং রাস্তার পাশে বহুতল ভবন ও বৈদ্যুতিক খুঁটি রয়েছে।")
    print(
        "Example 26 OUTPUT: "
        + " ".join(model.caption_image(test_img26.to(device), dataset.vocab))
    )

    test_img27 = transform(Image.open("test_examples/27.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 27 CORRECT: শহীদ মিনারে পুষ্পস্তবক অর্পণ করা হয়েছে পেছনে এবং পেছনে গাছপালা আছে।")
    print(
        "Example 27 OUTPUT: "
        + " ".join(model.caption_image(test_img27.to(device), dataset.vocab))
    )

    test_img28 = transform(Image.open("test_examples/28.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 28 CORRECT: স্মৃতিসৌধের সামনে অনেকজন মানুষ বাংলাদেশের পতাকা হাতে দাঁড়িয়ে রয়েছে এবং স্মৃতিসৌধের পেছনে গাছপালা আছে।")
    print(
        "Example 28 OUTPUT: "
        + " ".join(model.caption_image(test_img28.to(device), dataset.vocab))
    )

    test_img29 = transform(Image.open("test_examples/29.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 29 CORRECT: রাস্তার পাশে দাঁড়িয়ে একজন খাবার বিক্রেতা কয়েকজন মানুষের কাছে খাবার বিক্রি করছে ও ভবনে সাইনবোর্ড রয়েছে।")
    print(
        "Example 29 OUTPUT: "
        + " ".join(model.caption_image(test_img29.to(device), dataset.vocab))
    )

    test_img30 = transform(Image.open("test_examples/30.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 30 CORRECT: রাস্তা দিয়ে মানুষজন ও যানবাহন চলাচল করছে এবং রাস্তার পাশের ভবনে দোকানপাট ও সাইনবোর্ড রয়েছে।")
    print(
        "Example 30 OUTPUT: "
        + " ".join(model.caption_image(test_img30.to(device), dataset.vocab))
    )

    test_img31 = transform(Image.open("test_examples/31.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 31 CORRECT: ফুল গাছের ডালে একটি পাখি বসে আছে।")
    print(
        "Example 31 OUTPUT: "
        + " ".join(model.caption_image(test_img31.to(device), dataset.vocab))
    )

    test_img32 = transform(Image.open("test_examples/32.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 32 CORRECT: কয়েকজন কৃষক ফসলের জমি থেকে ধান কাটছে।")
    print(
        "Example 32 OUTPUT: "
        + " ".join(model.caption_image(test_img32.to(device), dataset.vocab))
    )

    test_img33 = transform(Image.open("test_examples/33.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 33 CORRECT: একটি বাঁশ বাগান দেখা যাচ্ছে।")
    print(
        "Example 33 OUTPUT: "
        + " ".join(model.caption_image(test_img33.to(device), dataset.vocab))
    )

    test_img34 = transform(Image.open("test_examples/34.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 34 CORRECT: রাস্তার পাশে গাছপালা ও ফসলের জমি আছে।")
    print(
        "Example 34 OUTPUT: "
        + " ".join(model.caption_image(test_img34.to(device), dataset.vocab))
    )

    test_img35 = transform(Image.open("test_examples/35.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 35 CORRECT: রাস্তা দিয়ে যানবাহন চলাচল করছে এবং রাস্তার পাশের ভবনে দোকান ও সাইনবোর্ড রয়েছে।")
    print(
        "Example 35 OUTPUT: "
        + " ".join(model.caption_image(test_img35.to(device), dataset.vocab))
    )

    test_img36 = transform(Image.open("test_examples/36.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 36 CORRECT: একটি দোকানের সামনে একজন মানুষ দাঁড়িয়ে আছে এবং দোকানে অনেক ধরনের পণ্য সামগ্রী রয়েছে।")
    print(
        "Example 36 OUTPUT: "
        + " ".join(model.caption_image(test_img36.to(device), dataset.vocab))
    )

    test_img37 = transform(Image.open("test_examples/37.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 37 CORRECT: করোনা মহামারীর সময় মাস্ক পরিধান করে কয়েকজন মানুষ বাজারে সবজি কেনাবেচা করছে।")
    print(
        "Example 37 OUTPUT: "
        + " ".join(model.caption_image(test_img37.to(device), dataset.vocab))
    )

    test_img38 = transform(Image.open("test_examples/38.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 38 CORRECT: ফিলিং স্টেশনে কয়েকটি গাড়ি সারিবদ্ধ ভাবে দাঁড়িয়ে রয়েছে।")
    print(
        "Example 38 OUTPUT: "
        + " ".join(model.caption_image(test_img38.to(device), dataset.vocab))
    )

    test_img39 = transform(Image.open("test_examples/39.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 39 CORRECT: চা বাগান থেকে অনেক জন মহিলা চা পাতা সংগ্রহ করছে।")
    print(
        "Example 39 OUTPUT: "
        + " ".join(model.caption_image(test_img39.to(device), dataset.vocab))
    )

    test_img40 = transform(Image.open("test_examples/40.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 40 CORRECT: অনেক জন মানুষ গাছের নিচে সারিবদ্ধ ভাবে দুটি ব্যানার হাতে দাঁড়িয়ে রয়েছে এবং পেছনে ভবন ও সাইনবোর্ড আছে।")
    print(
        "Example 40 OUTPUT: "
        + " ".join(model.caption_image(test_img40.to(device), dataset.vocab))
    )

    test_img41 = transform(Image.open("test_examples/41.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 41 CORRECT: বর্ষার মৌসুমে বৃষ্টির সময় জলাশয়ের পাড়ে দুজন মানুষ ছাতা মাথায় দিয়ে বসে রয়েছে।")
    print(
        "Example 41 OUTPUT: "
        + " ".join(model.caption_image(test_img41.to(device), dataset.vocab))
    )

    test_img42 = transform(Image.open("test_examples/42.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 42 CORRECT: একজন চা বিক্রেতা চা দোকানে বসে কেটলি থেকে চা বানাচ্ছে।")
    print(
        "Example 42 OUTPUT: "
        + " ".join(model.caption_image(test_img42.to(device), dataset.vocab))
    )

    test_img43 = transform(Image.open("test_examples/43.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 43 CORRECT: কয়েকজন নৃত্যশিল্পী মঞ্চে নৃত্য প্রদর্শন করছে।")
    print(
        "Example 43 OUTPUT: "
        + " ".join(model.caption_image(test_img43.to(device), dataset.vocab))
    )

    test_img44 = transform(Image.open("test_examples/44.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 44 CORRECT: বর্ষার মৌসুমে বৃষ্টির সময় একজন মহিলা ছাতা মাথায় দিয়ে রাস্তা দিয়ে হেটে যাচ্ছে এবং রাস্তার পাশে দোকান আছে।")
    print(
        "Example 44 OUTPUT: "
        + " ".join(model.caption_image(test_img44.to(device), dataset.vocab))
    )

    test_img45 = transform(Image.open("test_examples/45.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 45 CORRECT: গ্রামীণ মাটির রাস্তার দুই পাশে গাছপালা ও ফসলের জমি আছে।")
    print(
        "Example 45 OUTPUT: "
        + " ".join(model.caption_image(test_img45.to(device), dataset.vocab))
    )

    test_img46 = transform(Image.open("test_examples/46.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 46 CORRECT: রেস্টুরেন্টের চেয়ারে কয়েকজন মানুষ বসে আছে এবং একজন ব্যক্তি টেবিলে খাবার সাজিয়ে রেখেছে।")
    print(
        "Example 46 OUTPUT: "
        + " ".join(model.caption_image(test_img46.to(device), dataset.vocab))
    )

    test_img47 = transform(Image.open("test_examples/47.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 47 CORRECT: বাংলাদেশ সেনাবাহিনী কয়েকজন সদস্য বন্দুক  হাতে সারিবদ্ধ ভাবে প্যারেড করছে।")
    print(
        "Example 47 OUTPUT: "
        + " ".join(model.caption_image(test_img47.to(device), dataset.vocab))
    )

    test_img48 = transform(Image.open("test_examples/48.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 48 CORRECT: কয়েকজন বিজিবির সদস্য একটি মাঠের মধ্যে দাঁড়িয়ে কথা বলছে এবং পেছনে ফসলের জমি ও গাছপালা আছে।")
    print(
        "Example 48 OUTPUT: "
        + " ".join(model.caption_image(test_img48.to(device), dataset.vocab))
    )
    
    test_img49 = transform(Image.open("test_examples/49.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 49 CORRECT: শীতের সকালে রাস্তা দিয়ে যানবাহন মানুষজন চলাচল করছে এবং রাস্তার পাশে গাছপালা আছে।")
    print(
        "Example 49 OUTPUT: "
        + " ".join(model.caption_image(test_img49.to(device), dataset.vocab))
    )
    
    test_img50 = transform(Image.open("test_examples/50.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 50 CORRECT: কয়েকজন উপজাতি মেয়ে একটি স্কুল মাঠে নিত্য প্রদর্শন করছে এবং পেছনে ভবন ও গাছপালা রয়েছে।")
    print(
        "Example 50 OUTPUT: "
        + " ".join(model.caption_image(test_img50.to(device), dataset.vocab))
    )

    
    

    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
