

class StudySchema(schema.schema):
    def __init__(self, schema_dict):
        super().__init__(schema_dict)

        # Ordered list of the fields making up names
        self.field_names = [
            key if instance(key,str) else key.schema
            for key in schema_dict]
        self.name_format = '{' + '}_{'.join(self.field_names) + '}'

    def name(study_args):
        """Produces a canonical name for the study.
        study_args: dict
            Values parsed against schema
        """
        formatted = {
            k:field.format(study_args[k])
            for k,field in schemautil.schema_items(self.schema)}
        return self.name_format.format(**formatted)

    def parse(name):
        """Parses a name back into a validated study_args dict"""
        raw = dict(zip(self.field_names, '_'.split(name)))
        parsed = self.schema.validate(raw)
        return parsed




schemautil.schema_getitem(self.schema
