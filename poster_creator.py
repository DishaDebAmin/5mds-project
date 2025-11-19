from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random

class PosterCreator:
    def __init__(self):
        self.templates = {
            'modern': self._create_modern_poster,
            'creative': self._create_creative_poster,
            'minimal': self._create_minimal_poster,
            'bold': self._create_bold_poster
        }
    
    def create_social_media_poster(self, brand_name, tagline, ad_copy, logo_path, 
                                 primary_color, secondary_color, style='modern'):
        """Create a complete social media poster"""
        width, height = 1080, 1080  # Instagram square format
        poster = Image.new('RGB', (width, height), '#FFFFFF')
        draw = ImageDraw.Draw(poster)
        
        # Choose template based on style
        template_func = self.templates.get(style, self._create_modern_poster)
        template_func(draw, width, height, brand_name, tagline, ad_copy, 
                     primary_color, secondary_color)
        
        # Add logo if provided
        if logo_path and os.path.exists(logo_path.lstrip('/')):
            try:
                logo = Image.open(logo_path.lstrip('/')).convert('RGBA')
                logo = logo.resize((150, 150))
                poster.paste(logo, (width - 180, 30), logo)
            except Exception as e:
                print(f"Could not add logo: {e}")
        
        # Save poster
        filename = f"poster_{brand_name.replace(' ', '_').lower()}_{random.randint(1000,9999)}.png"
        filepath = os.path.join('static', 'generated', filename)
        poster.save(filepath, quality=95)
        
        return {
            'filename': filename,
            'path': f"/static/generated/{filename}",
            'style': style,
            'dimensions': f"{width}x{height}"
        }
    
    def _create_modern_poster(self, draw, width, height, brand_name, tagline, ad_copy, primary_color, secondary_color):
        """Modern poster template"""
        # Background gradient
        for i in range(height):
            ratio = i / height
            r = int(255 * (1 - ratio) + int(primary_color[1:3], 16) * ratio)
            g = int(255 * (1 - ratio) + int(primary_color[3:5], 16) * ratio)
            b = int(255 * (1 - ratio) + int(primary_color[5:7], 16) * ratio)
            draw.line([(0, i), (width, i)], fill=(r, g, b))
        
        # Content
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            tagline_font = ImageFont.truetype("arial.ttf", 36)
            body_font = ImageFont.truetype("arial.ttf", 28)
        except:
            title_font = ImageFont.load_default()
            tagline_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
        
        # Brand name
        draw.text((width//2, 200), brand_name, fill='#FFFFFF', font=title_font, anchor="mm")
        
        # Tagline
        draw.text((width//2, 300), tagline, fill='#FFFFFF', font=tagline_font, anchor="mm")
        
        # Ad copy
        lines = self._wrap_text(ad_copy['description'], body_font, width - 100)
        for i, line in enumerate(lines):
            draw.text((width//2, 450 + i*40), line, fill='#FFFFFF', font=body_font, anchor="mm")
        
        # CTA Button
        button_y = 700
        draw.rectangle([width//2-100, button_y, width//2+100, button_y+60], fill=secondary_color)
        draw.text((width//2, button_y+30), ad_copy['cta'], fill='#FFFFFF', font=body_font, anchor="mm")
    
    def _create_creative_poster(self, draw, width, height, brand_name, tagline, ad_copy, primary_color, secondary_color):
        """Creative poster template with geometric elements"""
        # Background
        draw.rectangle([0, 0, width, height], fill=primary_color)
        
        # Geometric shapes
        draw.ellipse([-100, -100, 400, 400], fill=secondary_color, outline=None)
        draw.rectangle([width-300, height-300, width+100, height+100], fill=secondary_color, outline=None)
        
        # Content
        try:
            title_font = ImageFont.truetype("arial.ttf", 64)
            tagline_font = ImageFont.truetype("arial.ttf", 32)
            body_font = ImageFont.truetype("arial.ttf", 24)
        except:
            title_font = ImageFont.load_default()
            tagline_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
        
        # Brand name
        draw.text((100, 150), brand_name, fill='#FFFFFF', font=title_font)
        
        # Tagline
        draw.text((100, 250), tagline, fill='#FFFFFF', font=tagline_font)
        
        # Ad copy
        lines = self._wrap_text(ad_copy['description'], body_font, width - 200)
        for i, line in enumerate(lines):
            draw.text((100, 350 + i*35), line, fill='#FFFFFF', font=body_font)
        
        # CTA
        draw.text((100, 600), ad_copy['cta'], fill=secondary_color, font=title_font)
    
    def _create_minimal_poster(self, draw, width, height, brand_name, tagline, ad_copy, primary_color, secondary_color):
        """Minimal poster template"""
        # Background
        draw.rectangle([0, 0, width, height], fill='#FFFFFF')
        
        # Accent bar
        draw.rectangle([0, 0, width, 100], fill=primary_color)
        
        # Content
        try:
            title_font = ImageFont.truetype("arial.ttf", 60)
            tagline_font = ImageFont.truetype("arial.ttf", 28)
            body_font = ImageFont.truetype("arial.ttf", 22)
        except:
            title_font = ImageFont.load_default()
            tagline_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
        
        # Brand name
        draw.text((width//2, 50), brand_name, fill='#FFFFFF', font=title_font, anchor="mm")
        
        # Tagline
        draw.text((width//2, 200), tagline, fill=primary_color, font=tagline_font, anchor="mm")
        
        # Ad copy
        lines = self._wrap_text(ad_copy['description'], body_font, width - 150)
        for i, line in enumerate(lines):
            draw.text((width//2, 300 + i*35), line, fill='#333333', font=body_font, anchor="mm")
        
        # CTA
        draw.text((width//2, 500), ad_copy['cta'], fill=secondary_color, font=title_font, anchor="mm")
    
    def _create_bold_poster(self, draw, width, height, brand_name, tagline, ad_copy, primary_color, secondary_color):
        """Bold and vibrant poster template"""
        # Background pattern
        for i in range(0, width, 50):
            for j in range(0, height, 50):
                if (i + j) % 100 == 0:
                    draw.rectangle([i, j, i+50, j+50], fill=primary_color)
                else:
                    draw.rectangle([i, j, i+50, j+50], fill=secondary_color)
        
        # Content box
        draw.rectangle([50, 50, width-50, height-50], fill='#FFFFFF')
        
        # Content
        try:
            title_font = ImageFont.truetype("arial.ttf", 68)
            tagline_font = ImageFont.truetype("arial.ttf", 34)
            body_font = ImageFont.truetype("arial.ttf", 26)
        except:
            title_font = ImageFont.load_default()
            tagline_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
        
        # Brand name
        draw.text((width//2, 150), brand_name, fill=primary_color, font=title_font, anchor="mm")
        
        # Tagline
        draw.text((width//2, 250), tagline, fill=secondary_color, font=tagline_font, anchor="mm")
        
        # Ad copy
        lines = self._wrap_text(ad_copy['description'], body_font, width - 200)
        for i, line in enumerate(lines):
            draw.text((width//2, 350 + i*40), line, fill='#333333', font=body_font, anchor="mm")
        
        # CTA
        draw.rectangle([width//2-120, 600, width//2+120, 660], fill=primary_color)
        draw.text((width//2, 630), ad_copy['cta'], fill='#FFFFFF', font=body_font, anchor="mm")
    
    def _wrap_text(self, text, font, max_width):
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line) if hasattr(font, 'getbbox') else font.getmask(test_line).getbbox()
            text_width = bbox[2] - bbox[0] if bbox else len(test_line) * 10
            
            if text_width <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines