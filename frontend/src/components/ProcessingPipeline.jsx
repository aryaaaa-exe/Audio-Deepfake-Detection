import { motion } from "framer-motion";
export default function ProcessingPipeline({ spectrogram }) {

    if (!spectrogram) return null;

    return (

        <div className="pipeline">

            <div className="spectrogram-container">

                <img
                    src={spectrogram}
                    alt="Mel Spectrogram"
                    className="spectrogram-image hidden-spectrogram"
                />

                <div className="patch-grid">

    {Array.from({ length: 160 }).map((_, i) => {

        const row = Math.floor(i / 32);
        const col = i % 32;

        return (
            <motion.div
    key={i}
    className="patch-cell"
    initial={{
        scale: 0.8,
        opacity: 0
    }}
    animate={{
        scale: 1,
        opacity: 1
    }}
    transition={{
        delay: row * 0.15 + col * 0.01,
        duration: 0.25
    }}
    style={{
        backgroundImage: `url(${spectrogram})`,
        backgroundSize: "3200% 500%",
        backgroundPosition: `${(col / 31) * 100}% ${(row / 4) * 100}%`
    }}
/>
        );

    })}

</div>
            </div>

        </div>

    );

}