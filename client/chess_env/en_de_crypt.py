import chess
from concrete.ml.deployment import FHEModelClient

class EnDe_crypt:
    def __init__(self) -> None:
        self.source = "client/source"
        self.target = "client/target"
        self.fhesource_client = FHEModelClient(path_dir=self.source, key_dir=self.source)
        self.fhesource_client.load()


    """
    input: float --> quantization --> encryption
    """
    def encrypt(self, clear_input):

        encrypted_input = self.fhesource_client.quantize_encrypt_serialize(clear_input)
        return encrypted_input
    
    """
    output: float <-- dequantization <-- decryption
    """
    def decrypt(self, encrypted_output):

        decrypted_data = self.fhesource_client.deserialize_decrypt_dequantize(encrypted_output)
        return decrypted_data



    def square_to_alpha(self, src_sq, trgt_sq):
        """
        convert square number into chessboard digit coordinates (due to chess lib) and alpha.
        input: source square number, target square number
        return : uci format (str), source and target as digit coordinates
        """

        # digit conversion
        col_s, row_s = chess.square_file(src_sq),chess.square_rank(src_sq)
        col_t, row_t = chess.square_file(trgt_sq),chess.square_rank(trgt_sq)

        # alpha conversion
        alpha_col_s = chess.FILE_NAMES[col_s]
        alpha_col_t = chess.FILE_NAMES[col_t]
        
        # converting coordinates to str and join to get uci move format (see chess lib)
        move_proposal = "".join((str(alpha_col_s),str(row_s+1),str(alpha_col_t),str(row_t+1)))

        return  move_proposal, ((col_s,row_s),(col_t,row_t))




